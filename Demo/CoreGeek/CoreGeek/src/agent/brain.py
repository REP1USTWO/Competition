"""Day/night strategy state machine for the competition agent.

Evidence levels: A = official rules, B = official samples/tests, C = strategy
inference, D = unconfirmed. The validator enforces A/B hard rules; everything
below is C-level policy and intentionally easy to tune.

Long-term loop: daytime build -> mine -> sell -> buy -> upgrade, with a dusk
return driven by remaining day rounds vs. path length; nighttime stable
controller assignment and weapon-specific targeting; per-round replanning with
feedback backoff so losses (role/tower/base damage) degrade gracefully across
many day/night cycles.
"""

import threading
from copy import deepcopy
from typing import Any, Iterable

from .grid import next_step, path_length
from .layout import Layout
from .protocol import (
    CONTROLLABLE_TYPES,
    MINERALS,
    PIONEER,
    Pos,
    Robot,
    Turn,
    Unit,
    WEAPON_BUILD_COST,
    WORKER,
    attack_command,
    build_command,
    buy_command,
    collect_command,
    distance,
    move_command,
    sell_command,
    station_footprint,
    use_command,
)
from .validator import TurnPlanningContext

DUSK_MARGIN = 6
SITE_FAILURE_LIMIT = 2
MOVE_BLOCK_ROUNDS = 3
MINE_RETRY_ROUNDS = 5
MAX_TOWERS = 3
WALL_START_DAY = 3
WALL_GATE_COUNT = 2
WALL_MATERIAL = "stone"
TOWER_MAX_HP = {1: 1000, 2: 1500, 3: 2000}


class StrategyConfig:
    """Tunable C-level policy knobs; defaults are the frozen Champion."""

    def __init__(
        self,
        loadout: tuple[str, ...] | None = None,
        day1_tower_cap: int = 3,
        wall_mode: str = "ring",            # ring | none
        wall_start_day: int = WALL_START_DAY,
        dusk_margin: int = DUSK_MARGIN,
        sell_fill_fraction: float = 1.0,    # bank once backpack fill >= fraction
        gold_reserve: int = 0,
        upgrade_use_threshold: float = 1.0,  # use voucher when hp fraction <= this
        consumables_enabled: bool = False,
        consumable_name: str = "DizzyWeapon",
        consumable_from_day: int = 6,
        consumable_min_cluster: float = 8.0,
        wall_workers: int = 1,
        station_l3_first: bool = True,
    ) -> None:
        self.loadout = loadout
        self.day1_tower_cap = day1_tower_cap
        self.wall_mode = wall_mode
        self.wall_start_day = wall_start_day
        self.dusk_margin = dusk_margin
        self.sell_fill_fraction = sell_fill_fraction
        self.gold_reserve = gold_reserve
        self.upgrade_use_threshold = upgrade_use_threshold
        self.consumables_enabled = consumables_enabled
        self.consumable_name = consumable_name
        self.consumable_from_day = consumable_from_day
        self.consumable_min_cluster = consumable_min_cluster
        self.wall_workers = wall_workers
        self.station_l3_first = station_l3_first


def champion_config() -> StrategyConfig:
    """Frozen final selection: validation2.txt favors two towers on day one.

    StrategyConfig defaults retain the historical rush baseline so archived
    experiment names remain reproducible. Production explicitly selects here.
    """
    return StrategyConfig(day1_tower_cap=2)


_CONFIG = champion_config()


def set_config(config: StrategyConfig) -> None:
    global _CONFIG
    _CONFIG = config


def get_config() -> StrategyConfig:
    return _CONFIG
DEFAULT_MINERAL_PRICE = {"stone": 1, "iron": 3, "copper": 5}
ROBOT_VALUE = {"smallRobot": 1, "middleRobot": 2, "largeRobot": 4, "bossRobot": 10}
_NEIGHBOUR_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


class Memory:
    """Per-match learned state; resets on a new match or half (round restart)."""

    def __init__(self) -> None:
        self.team_key: tuple[str, str] | None = None
        self.station_pos: Pos | None = None
        self.last_round = 0
        self.last_turn: Turn | None = None
        self.assignment: dict[int, int] = {}        # role_id -> tower_id
        self.last_commands: dict[int, dict[str, Any]] = {}
        self.build_failures: dict[tuple[Pos, str], int] = {}
        self.temp_blocked: dict[Pos, int] = {}      # cell -> usable again after round
        self.mine_cooldown: dict[Pos, int] = {}     # mine -> retry after round
        self.weapon_sites: set[Pos] = set()         # observed legal weapon cells
        self.focus_mineral: dict[int, str] = {}     # worker_id -> current trip mineral

    def reset(self) -> None:
        self.__init__()


_LOCK = threading.Lock()
_MEMORY = Memory()
_LAYOUT: Layout | None = None


def reset_memory() -> None:
    with _LOCK:
        _MEMORY.reset()


def _layout() -> Layout:
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = Layout()
    return _LAYOUT


def decide(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    turn = Turn.load(payload)
    with _LOCK:
        if _MEMORY.last_turn == turn:
            return {str(k): deepcopy(v) for k, v in _MEMORY.last_commands.items()}
        mem = _session(turn, _MEMORY)
        _review_feedback(turn, mem)
        _observe(turn, mem)
        state = _State(turn, mem)
        commands: dict[int, dict[str, Any]] = {}
        if turn.is_day:
            _day(state, commands)
        else:
            _night(state, commands)
        context = TurnPlanningContext(turn)
        for key, value in commands.items():
            context.try_add(key, value)
        result = context.dump()
        mem.last_commands = {int(key): value for key, value in result.items()}
        mem.last_round = turn.round_no
        mem.last_turn = turn
        return result


def _session(turn: Turn, mem: Memory) -> Memory:
    team_key = (turn.team_id, turn.team_type)
    station = turn.station()
    if station is not None:
        mem.station_pos = station.pos
    if mem.team_key != team_key or turn.round_no <= mem.last_round:
        station_pos = mem.station_pos if mem.team_key == team_key else None
        mem.reset()
        mem.team_key = team_key
        mem.station_pos = station.pos if station is not None else station_pos
    return mem


def _review_feedback(turn: Turn, mem: Memory) -> None:
    """Deterministic backoff: never blindly repeat a failed action (B: feedback field)."""
    if turn.round_no != mem.last_round + 1:
        return  # Feedback describes the immediately preceding round only.
    for unit_id, ok in turn.last_results.items():
        command = mem.last_commands.get(unit_id)
        if ok:
            continue
        if command is None:
            continue
        action = command.get("action")
        targets = command.get("targetPos") or []
        pos = None
        if targets and isinstance(targets[0], dict):
            try:
                pos = Pos(int(targets[0]["x"]), int(targets[0]["y"]))
            except (KeyError, TypeError, ValueError):
                pos = None
        if action == "move" and pos is not None:
            mem.temp_blocked[pos] = turn.round_no + MOVE_BLOCK_ROUNDS
        elif action == "build" and pos is not None:
            key = (pos, str(command.get("name")))
            mem.build_failures[key] = mem.build_failures.get(key, 0) + 1
        elif action == "collect" and pos is not None:
            mem.mine_cooldown[pos] = turn.round_no + MINE_RETRY_ROUNDS


def _observe(turn: Turn, mem: Memory) -> None:
    # A standing weapon is positive proof (B) that its cell is a weapon site.
    mem.weapon_sites.update(tower.pos for tower in turn.weapons())
    mem.temp_blocked = {p: r for p, r in mem.temp_blocked.items() if r >= turn.round_no}
    mem.mine_cooldown = {p: r for p, r in mem.mine_cooldown.items() if r >= turn.round_no}


class _State:
    """Per-turn derived view with cached path lengths."""

    def __init__(self, turn: Turn, mem: Memory) -> None:
        self.turn = turn
        self.mem = mem
        self.station = turn.station()
        self.footprint = station_footprint(self.station.pos) if self.station else ()
        self.roles = {role.unit_id: role for role in turn.controllable()}
        self.towers = {tower.unit_id: tower for tower in turn.weapons()}
        self.claimed: set[Pos] = set()
        self._reach_cache: dict[tuple[int, Pos], int | None] = {}
        self.pairs: dict[int, int] = {}     # tower_id -> role_id
        self.posts: dict[int, Pos] = {}     # tower_id -> night stand cell
        self.role_post: dict[int, Pos] = {}  # role_id -> night stand cell

    def extra_blocked(self) -> set[Pos]:
        return set(self.mem.temp_blocked)

    def reach(self, role: Unit, goal: Pos) -> int | None:
        key = (role.unit_id, goal)
        if key not in self._reach_cache:
            self._reach_cache[key] = path_length(
                self.turn, role, goal, self.extra_blocked(),
            )
        return self._reach_cache[key]


# ---------------------------------------------------------------------------
# Shared geometry helpers


def _neighbours(pos: Pos) -> tuple[Pos, ...]:
    return tuple(Pos(pos.x + dx, pos.y + dy) for dx, dy in _NEIGHBOUR_STEPS)


def _footprint_distance(pos: Pos, footprint: tuple[Pos, ...]) -> int:
    if not footprint:
        return 0
    return min(distance(pos, cell) for cell in footprint)


def _stand_cells(state: _State, target: Pos, role: Unit | None = None) -> list[Pos]:
    blocked = state.turn.blocked(role) if role else state.turn.occupied_cells()
    cells = [
        pos for pos in _neighbours(target)
        if state.turn.land(pos) and pos not in blocked
    ]
    cells.sort(key=lambda pos: (
        _footprint_distance(pos, state.footprint),
        distance(role.pos, pos) if role else 0,
        pos.x, pos.y,
    ))
    return cells


def _approach(
    state: _State,
    role: Unit,
    target: Pos,
    claimed: set[Pos],
    *,
    adjacent: bool = True,
) -> Pos | None:
    """One step toward target (or a cell adjacent to it); None if arrived/stuck."""
    if adjacent and distance(role.pos, target) <= 1:
        return None
    if not adjacent and role.pos == target:
        return None
    goals = _stand_cells(state, target, role) if adjacent else [target]
    for goal in goals:
        if goal in claimed:
            continue
        step = next_step(state.turn, role, goal, state.extra_blocked())
        if step is None or step in claimed:
            continue
        claimed.add(step)
        return step
    return None


# ---------------------------------------------------------------------------
# Controller assignment (stable across rounds and day/night cycles)


def _assignment(state: _State) -> None:
    turn, mem = state.turn, state.mem
    posts: dict[int, Pos] = {}
    for tower_id, tower in state.towers.items():
        cells = _stand_cells(state, tower.pos)
        if cells:
            posts[tower_id] = cells[0]
        else:
            # An occupied operating position is usable by its current controller.
            # Other occupants still remain obstacles to approaching characters.
            nearby = [role for role in state.roles.values()
                      if distance(role.pos, tower.pos) == 1]
            if nearby:
                nearby.sort(key=lambda role: (mem.assignment.get(role.unit_id) != tower_id,
                                               role.unit_id))
                posts[tower_id] = nearby[0].pos
    pairs: dict[int, int] = {}
    used: set[int] = set()
    # 1. Keep still-valid pairings: both alive, role can stand by the tower.
    for role_id, tower_id in sorted(mem.assignment.items()):
        role, tower = state.roles.get(role_id), state.towers.get(tower_id)
        post = posts.get(tower_id)
        if role is None or tower is None or post is None:
            continue
        if tower_id in pairs or role_id in used:
            continue
        if distance(role.pos, tower.pos) <= 1 or state.reach(role, post) is not None:
            pairs[tower_id] = role_id
            used.add(role_id)
    # 2. Fill the rest greedily by shortest path; unreachable pairs are banned.
    candidates = []
    for tower_id, tower in state.towers.items():
        if tower_id in pairs:
            continue
        post = posts.get(tower_id)
        if post is None:
            continue
        for role_id, role in state.roles.items():
            if role_id in used:
                continue
            if distance(role.pos, tower.pos) <= 1:
                cost = 0
            else:
                reach = state.reach(role, post)
                cost = reach if reach is not None else None
            if cost is not None:
                candidates.append((cost, role_id, tower_id))
    for _, role_id, tower_id in sorted(candidates):
        if tower_id in pairs or role_id in used:
            continue
        pairs[tower_id] = role_id
        used.add(role_id)
    mem.assignment = {role_id: tower_id for tower_id, role_id in pairs.items()}
    state.pairs = pairs
    state.posts = posts
    for tower_id, role_id in pairs.items():
        post = posts.get(tower_id)
        if post is not None:
            state.role_post[role_id] = post
    # Unassigned roles hold next to the station.
    if state.station is not None:
        station_cells = [
            pos for cell in state.footprint for pos in _neighbours(cell)
            if state.turn.land(pos) and pos not in state.footprint
        ]
        station_cells.sort(key=lambda pos: (pos.x, pos.y))
        index = 0
        for role_id, role in sorted(state.roles.items()):
            if role_id in state.role_post:
                continue
            while index < len(station_cells) and station_cells[index] in state.turn.occupied_cells():
                index += 1
            if index < len(station_cells):
                state.role_post[role_id] = station_cells[index]
                index += 1


# ---------------------------------------------------------------------------
# Day


def _day(state: _State, commands: dict[int, dict[str, Any]]) -> None:
    turn = state.turn
    if state.station is None:
        return
    _assignment(state)
    returning = _returning_roles(state)
    _build_phase(state, returning, commands)
    for role_id, role in sorted(state.roles.items()):
        if role_id in commands:
            continue
        if role_id in returning:
            _return_home(state, role, commands)
        elif role.kind == WORKER:
            _worker_day(state, role, commands)
        else:
            _pioneer_day(state, role, commands)


def _returning_roles(state: _State) -> set[int]:
    """Dusk rule (A: 70/60 rounds): head back before the walk home eats the margin."""
    remaining = state.turn.remaining_day_rounds
    returning: set[int] = set()
    for role_id, role in state.roles.items():
        post = state.role_post.get(role_id)
        if post is None:
            continue
        if role.pos == post:
            if remaining <= _CONFIG.dusk_margin:
                returning.add(role_id)
            continue
        walk = state.reach(role, post)
        if walk is None:
            # Unreachable right now: keep retrying rather than stranding far away.
            if remaining <= _CONFIG.dusk_margin * 2:
                returning.add(role_id)
            continue
        if remaining <= walk + _CONFIG.dusk_margin:
            returning.add(role_id)
    return returning


def _return_home(state: _State, role: Unit, commands: dict[int, dict[str, Any]]) -> None:
    turn = state.turn
    post = state.role_post.get(role.unit_id)
    if post is None:
        return
    if role.pos == post:
        _bank_at_vendor(state, role, commands, allow_detour=False)
        return
    # Bank carried minerals only when the vendor detour still beats the dusk line.
    minerals = [item for item in role.backpack if item in MINERALS]
    if minerals and _near_zone(state, role, "vendor") is not None:
        _bank_at_vendor(state, role, commands, allow_detour=False)
        return
    if minerals:
        vendor = _nearest_zone(state, role, "vendor")
        if vendor is not None:
            walk_home = state.reach(role, post)
            stand = _stand_cells(state, vendor, role)
            detour = None
            if stand:
                to_vendor = state.reach(role, stand[0])
                if to_vendor is not None:
                    back = path_length_at(state, stand[0], post)
                    if back is not None:
                        detour = to_vendor + back
            if (detour is not None and walk_home is not None
                    and turn.remaining_day_rounds > detour + DUSK_MARGIN + 2
                    and _mineral_value(state, minerals) >= 15):
                step = _approach(state, role, vendor, state.claimed)
                if step is not None:
                    commands[role.unit_id] = move_command(step)
                return
    step = _approach(state, role, post, state.claimed, adjacent=False)
    if step is not None:
        commands[role.unit_id] = move_command(step)


def path_length_at(state: _State, start: Pos, goal: Pos) -> int | None:
    probe = Unit(-1, start, WORKER, 1, 1, 0, 0, None, ())
    return path_length(state.turn, probe, goal, state.extra_blocked())


def _weapon_sites(state: _State) -> tuple[Pos, ...]:
    """Candidate weapon cells: verified layout, observed sites, ring fallback (C/D)."""
    turn = state.turn
    ordered: list[Pos] = []
    for pos in _layout().sites(turn, "weapons"):
        if pos not in ordered:
            ordered.append(pos)
    for pos in sorted(state.mem.weapon_sites, key=lambda p: (p.x, p.y)):
        if pos not in ordered:
            ordered.append(pos)
    if state.station is not None:
        ring = [
            pos for cell in state.footprint for pos in _neighbours(cell)
            if state.turn.land(pos) and pos not in state.footprint
        ]
        ring.sort(key=lambda pos: (pos.x, pos.y))
        for pos in ring:
            if pos not in ordered:
                ordered.append(pos)
    return tuple(ordered)


def _build_phase(
    state: _State,
    returning: set[int],
    commands: dict[int, dict[str, Any]],
) -> None:
    turn = state.turn
    standing = {tower.pos for tower in state.towers.values()}
    tower_cap = MAX_TOWERS
    if turn.day_number == 1:
        tower_cap = min(tower_cap, _CONFIG.day1_tower_cap)
    if len(standing) >= tower_cap or turn.gold < WEAPON_BUILD_COST:
        return
    loadout = _CONFIG.loadout or _layout().loadout
    sites = []
    for index, site in enumerate(_weapon_sites(state)):
        if site in standing:
            continue
        name = loadout[index % len(loadout)]
        if state.mem.build_failures.get((site, name), 0) >= SITE_FAILURE_LIMIT:
            continue
        if site in turn.occupied_cells() or site in state.claimed:
            continue
        sites.append((site, name))
    if not sites:
        return
    workers = [
        role for role in state.roles.values()
        if role.kind == WORKER and role.unit_id not in returning
        and role.unit_id not in commands
    ]
    slots = tower_cap - len(standing)
    # Only the top-priority candidates are real targets; fallback ring cells
    # enter the list only when better sites get blacklisted by failures.
    wanted = sites[:slots]
    pending = []
    # Pass 1: build immediately at sites a free worker already touches.
    for site, name in wanted:
        if not workers:
            break
        adjacent = [role for role in workers if distance(role.pos, site) <= 1]
        if not adjacent:
            pending.append((site, name))
            continue
        role = min(adjacent, key=lambda item: item.unit_id)
        commands[role.unit_id] = build_command(site, name)
        state.claimed.add(site)
        workers.remove(role)
        slots -= 1
    # Pass 2: send the nearest remaining worker toward each leftover site,
    # unless the trip already breaks the dusk line.
    for site, name in pending:
        if slots <= 0 or not workers:
            break
        workers.sort(key=lambda role: (distance(role.pos, site), role.unit_id))
        role = workers[0]
        post = state.role_post.get(role.unit_id)
        stand = _stand_cells(state, site, role)
        walk = state.reach(role, stand[0]) if stand else None
        if walk is not None and post is not None:
            back = path_length_at(state, stand[0], post)
            if back is not None and turn.remaining_day_rounds <= walk + back + DUSK_MARGIN:
                continue
        step = _approach(state, role, site, state.claimed)
        if step is not None:
            commands[role.unit_id] = move_command(step)
            workers.pop(0)
            slots -= 1


def _worker_day(state: _State, role: Unit, commands: dict[int, dict[str, Any]]) -> None:
    if role.health * 2 <= 220 and "Medicine" in role.backpack:
        commands[role.unit_id] = use_command("Medicine")
        return
    if _wall_duty(state, role, commands):
        return
    minerals = [item for item in role.backpack if item in MINERALS]
    fill_full = role.backpack_full or (
        role.capacity is not None
        and len(role.backpack) >= role.capacity * _CONFIG.sell_fill_fraction
    )
    if minerals and (fill_full or _should_bank(state, role, minerals)):
        if _bank_at_vendor(state, role, commands):
            return
    _mine(state, role, commands)


# ---------------------------------------------------------------------------
# Walls (C policy): a distance-2 ring soaks robot attacks while towers shoot.
# Legality of the wall zone is D-level; failed probes are blacklisted by the
# shared build-failure mechanism, so guessing costs at most a few rounds.


def _wall_sites(state: _State) -> tuple[Pos, ...]:
    if state.station is None:
        return ()
    towers = [tower.pos for tower in state.towers.values()]
    standing = {wall.pos for wall in state.turn.walls()}
    xs = [pos.x for pos in state.footprint]
    ys = [pos.y for pos in state.footprint]
    centre = Pos(state.turn.width // 2, state.turn.height // 2)
    ring = []
    for x in range(min(xs) - 2, max(xs) + 3):
        for y in range(min(ys) - 2, max(ys) + 3):
            pos = Pos(x, y)
            if not state.turn.land(pos):
                continue
            if _footprint_distance(pos, state.footprint) != 2:
                continue
            if any(distance(pos, tower) <= 1 for tower in towers):
                continue
            ring.append(pos)
    # Build the side facing the map centre first; the far side keeps gates so
    # roles can still leave for the economy and return at dusk.
    ring.sort(key=lambda pos: (distance(pos, centre), pos.x, pos.y))
    walled = ring[:-WALL_GATE_COUNT] if len(ring) > WALL_GATE_COUNT else ring
    return tuple(
        pos for pos in walled
        if pos not in standing
        and pos not in state.turn.occupied_cells()
        and state.mem.build_failures.get((pos, "wall"), 0) < SITE_FAILURE_LIMIT
    )


def _walls_pending(state: _State) -> bool:
    if _CONFIG.wall_mode == "none":
        return False
    if state.station is None or len(state.towers) < MAX_TOWERS:
        return False
    if not _layout().build_walls and state.turn.day_number < _CONFIG.wall_start_day:
        return False
    return bool(_wall_sites(state))


def _stone_worker_ids(state: _State) -> set[int]:
    workers = sorted(
        role.unit_id for role in state.roles.values() if role.kind == WORKER
    )
    return set(workers[: max(1, _CONFIG.wall_workers)])


def _wall_duty(state: _State, role: Unit, commands: dict[int, dict[str, Any]]) -> bool:
    if not _walls_pending(state) or role.unit_id not in _stone_worker_ids(state):
        return False
    sites = _wall_sites(state)
    if WALL_MATERIAL in role.backpack:
        adjacent = [pos for pos in sites if distance(role.pos, pos) == 1]
        if adjacent:
            commands[role.unit_id] = build_command(adjacent[0], "wall")
            state.claimed.add(adjacent[0])
            return True
        options = []
        for pos in sites:
            cells = _stand_cells(state, pos, role)
            if cells:
                walk = state.reach(role, cells[0])
                if walk is not None:
                    options.append((walk, pos))
        if options:
            options.sort(key=lambda item: (item[0], item[1].x, item[1].y))
            walk, pos = options[0]
            if state.turn.remaining_day_rounds > walk * 2 + DUSK_MARGIN + 2:
                step = _approach(state, role, pos, state.claimed)
                if step is not None:
                    commands[role.unit_id] = move_command(step)
                    return True
        return True
    state.mem.focus_mineral[role.unit_id] = WALL_MATERIAL
    _mine(state, role, commands)
    return True


def _should_bank(state: _State, role: Unit, minerals: list[str]) -> bool:
    turn = state.turn
    value = _mineral_value(state, minerals)
    towers_missing = len(state.towers) < MAX_TOWERS
    if towers_missing and turn.gold < WEAPON_BUILD_COST and value >= WEAPON_BUILD_COST - turn.gold:
        return True
    if turn.remaining_day_rounds <= 25 and value >= 20:
        return True
    return False


def _mineral_value(state: _State, minerals: Iterable[str]) -> int:
    prices = state.turn.vendor_prices
    return sum(prices.get(item, DEFAULT_MINERAL_PRICE.get(item, 1)) for item in minerals)


def _nearest_zone(state: _State, role: Unit, kind: str) -> Pos | None:
    zones = [pos for pos, zone in state.turn.zones.items() if zone == kind]
    if not zones:
        return None
    zones.sort(key=lambda pos: (distance(role.pos, pos), pos.x, pos.y))
    return zones[0]


def _near_zone(state: _State, role: Unit, kind: str) -> Pos | None:
    zones = [
        pos for pos, zone in state.turn.zones.items()
        if zone == kind and distance(role.pos, pos) == 1
    ]
    return zones[0] if zones else None


def _bank_at_vendor(
    state: _State,
    role: Unit,
    commands: dict[int, dict[str, Any]],
    *,
    allow_detour: bool = True,
) -> bool:
    vendor = _near_zone(state, role, "vendor")
    if vendor is not None:
        stacks: dict[str, int] = {}
        for item in role.backpack:
            if item in MINERALS:
                stacks[item] = stacks.get(item, 0) + 1
        if not stacks:
            return True
        name = max(
            stacks,
            key=lambda item: (
                stacks[item] * state.turn.vendor_prices.get(
                    item, DEFAULT_MINERAL_PRICE.get(item, 1)),
                stacks[item],
            ),
        )
        commands[role.unit_id] = sell_command(name, stacks[name])
        state.mem.focus_mineral.pop(role.unit_id, None)
        return True
    if not allow_detour:
        return False
    vendor = _nearest_zone(state, role, "vendor")
    if vendor is None:
        return False
    step = _approach(state, role, vendor, state.claimed)
    if step is not None:
        commands[role.unit_id] = move_command(step)
        return True
    return False


def _mine(state: _State, role: Unit, commands: dict[int, dict[str, Any]]) -> None:
    turn = state.turn
    if role.backpack_full:
        _bank_at_vendor(state, role, commands)
        return
    focus = state.mem.focus_mineral.get(role.unit_id)
    best: tuple[float, str, Pos] | None = None
    for mineral in MINERALS:
        if focus is not None and mineral != focus:
            continue
        price = turn.vendor_prices.get(mineral, DEFAULT_MINERAL_PRICE.get(mineral, 1))
        mines = [
            pos for pos, zone in turn.zones.items()
            if zone == mineral and state.mem.mine_cooldown.get(pos, 0) < turn.round_no
        ]
        if not mines:
            continue
        adjacent = [pos for pos in mines if distance(role.pos, pos) == 1]
        if adjacent:
            score = float(price)
            pick = min(adjacent, key=lambda p: (p.x, p.y))
            candidate = (score + 1000.0, mineral, pick)
        else:
            stand_options = []
            for pos in sorted(mines, key=lambda p: (distance(role.pos, p), p.x, p.y))[:3]:
                cells = _stand_cells(state, pos, role)
                if cells:
                    walk = state.reach(role, cells[0])
                    if walk is not None:
                        stand_options.append((walk, cells[0], pos))
            if not stand_options:
                continue
            walk, _, pos = min(stand_options, key=lambda item: item[0])
            # Round trip: going out must still leave time to walk back home.
            if turn.remaining_day_rounds <= walk * 2 + DUSK_MARGIN + 4:
                continue
            candidate = (price / (walk + 2.0), mineral, pos)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        if focus is not None:
            state.mem.focus_mineral.pop(role.unit_id, None)
            _mine(state, role, commands)
        return
    _, mineral, mine = best
    state.mem.focus_mineral[role.unit_id] = mineral
    if distance(role.pos, mine) == 1:
        commands[role.unit_id] = collect_command(mine)
        return
    step = _approach(state, role, mine, state.claimed)
    if step is not None:
        commands[role.unit_id] = move_command(step)


# ---------------------------------------------------------------------------
# Pioneer: upgrade runner (buy -> deliver -> use), then hold near the base


def _upgrade_plan(state: _State) -> list[tuple[str, Unit]]:
    """Repair-via-upgrade: upgrading restores full HP (A). Rebuild fund first.

    Order (C): weapon L1->L2 (rocket first: range/multi-target scale best),
    station L2 (raw survival HP), rocket L3 (full map), station L3, then the
    remaining weapon L2->L3 upgrades.
    """
    plan: list[tuple[str, Unit]] = []
    towers = sorted(state.towers.values(), key=lambda t: (t.kind != "rocket", t.level, t.health, t.unit_id))
    for tower in towers:
        if tower.level == 1:
            plan.append(("WeaponUpgradeVoucher1", tower))
    if state.station is not None and state.station.level == 1:
        plan.append(("StationUpgradeVoucher1", state.station))
    for tower in towers:
        if tower.kind == "rocket" and tower.level == 2:
            plan.append(("WeaponUpgradeVoucher2", tower))
    if state.station is not None and state.station.level == 2 and _CONFIG.station_l3_first:
        plan.append(("StationUpgradeVoucher2", state.station))
    for tower in towers:
        if tower.kind != "rocket" and tower.level == 2:
            plan.append(("WeaponUpgradeVoucher2", tower))
    if state.station is not None and state.station.level == 2 and not _CONFIG.station_l3_first:
        plan.append(("StationUpgradeVoucher2", state.station))
    return plan


def _pioneer_day(state: _State, role: Unit, commands: dict[int, dict[str, Any]]) -> None:
    turn = state.turn
    if role.health * 2 <= 200 and "Medicine" in role.backpack:
        commands[role.unit_id] = use_command("Medicine")
        return
    plan = _upgrade_plan(state)
    # 1. A voucher in the backpack goes to its building. With a delayed-use
    # threshold the voucher doubles as an emergency heal (A: upgrade = full HP).
    for name, building in plan:
        if name in role.backpack:
            max_hp = (1500 * building.level if building.kind == "station"
                      else TOWER_MAX_HP.get(building.level, 1000))
            # Delayed use is a station-only heal reserve; weapon vouchers are
            # firepower and always apply immediately.
            if (building.kind == "station"
                    and building.health > max_hp * _CONFIG.upgrade_use_threshold):
                break  # hold the voucher as a strategic reserve
            footprint = state.turn.footprint(building)
            if min(distance(role.pos, cell) for cell in footprint) == 1:
                commands[role.unit_id] = use_command(name, building.pos)
                return
            step = _approach(state, role, building.pos, state.claimed)
            if step is not None:
                commands[role.unit_id] = move_command(step)
            return
    # 2. Buy the next voucher when the rebuild fund is covered and time allows.
    # Do not hoard: an unused voucher in the backpack is already a reserve.
    towers_missing = len(state.towers) < MAX_TOWERS
    reserve = _CONFIG.gold_reserve + (WEAPON_BUILD_COST if towers_missing else 0)
    holding_voucher = any(name in role.backpack for name, _ in plan)
    for name, _ in plan:
        if name in role.backpack:
            continue  # never double-buy a voucher already being held
        price = turn.shop_prices.get(name)
        if price is None or turn.gold - price < reserve:
            continue
        if role.backpack_full:
            break
        shop = _near_zone(state, role, "weaponShop")
        if shop is not None:
            if turn.remaining_day_rounds > DUSK_MARGIN + 2:
                commands[role.unit_id] = buy_command(name)
            return
        shop = _nearest_zone(state, role, "weaponShop")
        if shop is None:
            return
        stand = _stand_cells(state, shop, role)
        walk = state.reach(role, stand[0]) if stand else None
        post = state.role_post.get(role.unit_id)
        if walk is not None and post is not None:
            back = path_length_at(state, stand[0], post)
            if back is not None and turn.remaining_day_rounds <= walk + back + DUSK_MARGIN + 2:
                return
        step = _approach(state, role, shop, state.claimed)
        if step is not None:
            commands[role.unit_id] = move_command(step)
        return
    # 3. Consumable reserve for boss nights (H: DizzyWeapon/Bomb, used by the
    # rocket controller during its cooldown window).
    if not _CONFIG.consumables_enabled or turn.day_number < _CONFIG.consumable_from_day:
        return
    name = _CONFIG.consumable_name
    price = turn.shop_prices.get(name)
    if price is None or turn.gold - price < reserve:
        return
    if holding_voucher or name in role.backpack or role.backpack_full:
        return
    shop = _near_zone(state, role, "weaponShop")
    if shop is not None:
        commands[role.unit_id] = buy_command(name)
        return
    shop = _nearest_zone(state, role, "weaponShop")
    if shop is not None:
        step = _approach(state, role, shop, state.claimed)
        if step is not None:
            commands[role.unit_id] = move_command(step)


# ---------------------------------------------------------------------------
# Night


def _night(state: _State, commands: dict[int, dict[str, Any]]) -> None:
    _assignment(state)
    expected: dict[int, int] = {}  # robot_id -> damage already claimed this round
    robots_at = {robot.pos: robot for robot in state.turn.robots if robot.health > 0}
    for tower_id, role_id in sorted(state.pairs.items()):
        tower, role = state.towers[tower_id], state.roles[role_id]
        if distance(role.pos, tower.pos) <= 1:
            if tower.cooldown > 0:
                continue
            targets = _pick_targets(state, tower, expected)
            if targets is not None:
                commands[tower_id] = attack_command(role_id, targets)
                # Deconflict: later towers prefer robots not already dying.
                per_target = {"gatling": 10, "railgun": 10 * tower.level,
                              "rocket": 20}.get(tower.kind, 10)
                for pos in targets if isinstance(targets, list) else [targets]:
                    robot = robots_at.get(pos)
                    if robot is not None:
                        expected[robot.robot_id] = expected.get(robot.robot_id, 0) + per_target
            continue
        step = _approach(state, role, tower.pos, state.claimed)
        if step is not None:
            commands[role.unit_id] = move_command(step)
    # Roles without a tower stay close to the station as a reserve.
    if state.station is not None:
        for role_id, role in sorted(state.roles.items()):
            if role_id in commands or role_id in state.pairs.values():
                continue
            if _footprint_distance(role.pos, state.footprint) <= 2:
                continue
            post = state.role_post.get(role_id)
            if post is None:
                continue
            step = _approach(state, role, post, state.claimed, adjacent=False)
            if step is not None:
                commands[role.unit_id] = move_command(step)
    _consumable_phase(state, commands)


def _consumable_phase(state: _State, commands: dict[int, dict[str, Any]]) -> None:
    """Throw DizzyWeapon/Bomb with roles whose tower is cooling down (free
    action) or that are unassigned. Survival gate: defensive clusters only."""
    if not _CONFIG.consumables_enabled:
        return
    if state.turn.day_number < _CONFIG.consumable_from_day or not state.turn.robots:
        return
    name = _CONFIG.consumable_name
    idle_roles = []
    for tower_id, role_id in state.pairs.items():
        tower = state.towers[tower_id]
        if tower.cooldown > 0 and role_id not in commands:
            idle_roles.append(state.roles[role_id])
    for role_id, role in state.roles.items():
        if role_id not in state.pairs.values() and role_id not in commands:
            idle_roles.append(role)
    holders = [role for role in idle_roles if name in role.backpack]
    if not holders:
        return
    target = _best_cluster(state)
    if target is None:
        return
    commands[holders[0].unit_id] = use_command(name, target)


def _best_cluster(state: _State) -> Pos | None:
    """Highest-value 3x3 robot cluster close to the base (defensive use only)."""
    robots = [robot for robot in state.turn.robots if robot.health > 0]
    best: tuple[float, Pos] | None = None
    for centre in robots:
        if _footprint_distance(centre.pos, state.footprint) > 6:
            continue
        value = sum(
            ROBOT_VALUE.get(other.kind, 1)
            for other in robots
            if distance(centre.pos, other.pos) <= 1
        )
        if best is None or value > best[0]:
            best = (value, centre.pos)
    if best is not None and best[0] >= _CONFIG.consumable_min_cluster:
        return best[1]
    return None


def _threat(state: _State, robot: Robot, damage: int, expected: int = 0) -> float:
    """Explainable scoring: protect the base first, prefer kills, skip bystanders."""
    gap = _footprint_distance(robot.pos, state.footprint) if state.footprint else 10
    hp_left = robot.health - expected
    score = 1000.0 - gap * 20.0
    score += ROBOT_VALUE.get(robot.kind, 1) * 5.0
    if robot.target_team:
        score += 30.0 if robot.target_team == state.turn.team_type else -200.0
    if robot.abnormal_state == "dizzy":
        score -= 100.0
    if hp_left <= 0:
        score -= 500.0  # another tower is already expected to finish it
    elif hp_left <= damage:
        score += 40.0
    score -= robot.health * 0.01
    return score


def _within_90(origin: Pos, first: Pos, second: Pos) -> bool:
    ax, ay = first.x - origin.x, first.y - origin.y
    bx, by = second.x - origin.x, second.y - origin.y
    return ax * bx + ay * by >= 0


def _pick_targets(
    state: _State,
    tower: Unit,
    expected: dict[int, int] | None = None,
) -> Pos | list[Pos] | None:
    reach = tower.range_of_attack()
    damage = {"gatling": 10, "railgun": 10 * tower.level, "rocket": 20}.get(tower.kind, 10)
    expected = expected or {}
    robots = [
        robot for robot in state.turn.robots
        if robot.health > 0 and 0 < distance(tower.pos, robot.pos) <= reach
    ]
    if not robots:
        return None
    # A scoring penalty is not a survival gate: predicted kills are uncertain
    # (e.g. railgun interception). Keep reachable home/unknown threats first.
    defensive = [robot for robot in robots
                 if robot.target_team in ("", state.turn.team_type)
                 or (state.footprint and _footprint_distance(robot.pos, state.footprint) <= 3)]
    if defensive:
        robots = defensive
    robots.sort(
        key=lambda robot: _threat(state, robot, damage, expected.get(robot.robot_id, 0)),
        reverse=True,
    )
    count = 1 if tower.kind == "railgun" else tower.level
    chosen: list[Robot] = []
    if tower.kind == "gatling":
        for robot in robots:
            if len(chosen) >= count:
                break
            if all(_within_90(tower.pos, robot.pos, other.pos) for other in chosen):
                chosen.append(robot)
    else:
        chosen = robots[:count]
    if not chosen:
        return None
    targets = [robot.pos for robot in chosen]
    # A/B: target count must equal the weapon level; repeated landing is legal.
    while len(targets) < count:
        targets.append(targets[0])
    return targets[0] if count == 1 else targets
