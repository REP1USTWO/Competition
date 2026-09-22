"""Validate supported actions and reserve their resources before emitting them.

This is deliberately not a substitute for the judger. In particular, the supplied
map contains no construction-zone metadata. Optional verified sites can narrow
construction, but no guessed ring or team orientation is a legality rule here.
"""

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .protocol import (
    CONTROLLABLE_TYPES, TOWER_TYPES, WEAPON_BUILD_COST,
    Pos, Turn, Unit, distance,
)

MINERALS = frozenset(("stone", "iron", "copper"))
UPGRADES = {
    **{f"WeaponUpgradeVoucher{n}": (TOWER_TYPES, n) for n in (1, 2)},
    **{f"WallUpgradeVoucher{n}": (("wall",), n) for n in (1, 2)},
    **{f"StationUpgradeVoucher{n}": (("station",), n) for n in (1, 2)},
}
_FIELDS = {
    "move": {"action", "targetPos"},
    "attack": {"action", "targetPos", "controllerId"},
    "build": {"action", "targetPos", "name"},
    "remove": {"action", "targetPos"},
    "collect": {"action", "targetPos"},
    "buy": {"action", "name", "num"},
    "sell": {"action", "name", "num"},
    "use": {"action", "name", "targetPos"},
    "drop": {"action", "name"},
}


class _Invalid(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Invalid(reason)


class TurnPlanningContext:
    """Transactional, per-turn command collection; rejected actions reserve nothing.

    A controller's action is consumed by operating a weapon, even though the
    response map stores that command under the weapon ID. Sales never finance
    same-turn purchases: their settlement order is not specified by the rules.
    """

    def __init__(
        self, turn: Turn, *, verified_sites: Mapping[str, set[Pos]] | None = None,
    ) -> None:
        self.turn = turn
        self.commands: dict[int, dict[str, Any]] = {}
        self.reserved_cells: set[Pos] = set()
        self.remaining_gold = max(0, turn.gold)
        self.rejections: list[dict[str, Any]] = []
        self.used_actors: set[int] = set()
        self.used_controllers: set[int] = set()
        self._changed_buildings: set[int] = set()
        self._tower_count = len(turn.weapons())
        self._units = {unit.unit_id: unit for unit in turn.ours}
        self._verified_sites = verified_sites

    def try_add(self, actor_id: int, command: dict[str, Any]) -> bool:
        """Accept a command only after all checks finish, then commit reservations."""
        try:
            _require(type(actor_id) is int, "actor ID must be an integer")
            _require(isinstance(command, dict), "command must be an object")
            actor = self._units.get(actor_id)
            _require(actor is not None and actor.health > 0, "unknown or dead actor")
            _require(actor_id not in self.used_actors, "actor already has an action")
            action = command.get("action")
            _require(isinstance(action, str) and action in _FIELDS,
                     "unsupported action (task and summon workflows are disabled)")
            _require(set(command) <= _FIELDS[action], "unexpected command fields")
            # All validation below is read-only. Values are committed at the end.
            gold_cost = 0
            tower_delta = 0
            reserve: Pos | None = None
            controller_id: int | None = None
            changed_building: int | None = None

            if action == "attack":
                controller_id = self._attack(actor, command)
            else:
                _require(actor.kind in CONTROLLABLE_TYPES, "action requires a character")
                if action == "move":
                    reserve = self._single_target(command)
                    _require(distance(actor.pos, reserve) == 1, "move must be one cell")
                    _require(self.turn.land(reserve), "move target is not land")
                    _require(reserve not in self.turn.blocked(actor), "move target is occupied")
                    _require(reserve not in self.reserved_cells, "target already reserved")
                elif action == "build":
                    reserve, gold_cost, tower_delta, changed_building = self._build(actor, command)
                elif action == "collect":
                    target = self._single_target(command)
                    _require(actor.kind == "worker", "only workers collect")
                    _require(distance(actor.pos, target) == 1, "mine is not adjacent")
                    _require(self.turn.zones.get(target) in MINERALS, "target is not a mine")
                    _require(not self._full(actor), "backpack is full")
                    # Official rules explicitly allow multiple workers at one mine.
                elif action == "remove":
                    target = self._single_target(command)
                    _require(actor.kind == "worker", "only workers remove walls")
                    _require(distance(actor.pos, target) == 1, "wall is not adjacent")
                    walls = (*self.turn.ours, *getattr(self.turn, "enemies", ()))
                    wall = next((u for u in walls if u.kind == "wall" and u.health > 0
                                 and u.pos == target), None)
                    _require(wall is not None, "target is not a living wall")
                    changed_building = wall.unit_id
                elif action in ("buy", "sell"):
                    gold_cost = self._trade(actor, command, buying=action == "buy")
                elif action == "drop":
                    _require(self._name(command) in actor.backpack, "item not in backpack")
                elif action == "use":
                    changed_building = self._use(actor, command)

            _require(gold_cost <= self.remaining_gold, "insufficient unreserved gold")
            if changed_building is not None:
                _require(changed_building not in self._changed_buildings,
                         "building already changed by another action")
                _require(changed_building not in self.commands,
                         "cannot modify a building already attacking")
            normalized = deepcopy(command)
        except (ValueError, TypeError, AttributeError, KeyError, OverflowError) as exc:
            self.rejections.append({"actor_id": actor_id, "reason": str(exc)})
            return False

        self.commands[actor_id] = normalized
        self.used_actors.add(actor_id)
        self.remaining_gold -= gold_cost
        self._tower_count += tower_delta
        if reserve is not None:
            self.reserved_cells.add(reserve)
        if controller_id is not None:
            self.used_actors.add(controller_id)
            self.used_controllers.add(controller_id)
        if changed_building is not None:
            self._changed_buildings.add(changed_building)
        return True

    def validate(self, commands: Mapping[Any, Any]) -> dict[str, dict[str, Any]]:
        """Append a mapping of candidate commands and return accepted wire commands."""
        if not isinstance(commands, Mapping):
            self.rejections.append({"actor_id": None, "reason": "commands must be a mapping"})
            return self.dump()
        for actor_id, command in commands.items():
            if isinstance(actor_id, str) and actor_id.isascii() and actor_id.isdecimal():
                try:
                    actor_id = int(actor_id)
                except ValueError:
                    self.rejections.append({"actor_id": None, "reason": "invalid actor ID"})
                    continue
            self.try_add(actor_id, command)
        return self.dump()

    def dump(self) -> dict[str, dict[str, Any]]:
        return {str(actor_id): deepcopy(command) for actor_id, command in self.commands.items()}

    def _targets(self, command: dict[str, Any], count: int) -> list[Pos]:
        raw = command.get("targetPos")
        _require(isinstance(raw, list) and len(raw) == count, "wrong target count")
        targets = []
        for item in raw:
            _require(isinstance(item, dict) and set(item) == {"x", "y"},
                     "target must contain x and y")
            _require(type(item["x"]) is int and type(item["y"]) is int,
                     "target coordinates must be integers")
            target = Pos(item["x"], item["y"])
            _require(0 <= target.x < self.turn.width and 0 <= target.y < self.turn.height,
                     "target is outside map")
            targets.append(target)
        return targets

    def _single_target(self, command: dict[str, Any]) -> Pos:
        return self._targets(command, 1)[0]

    @staticmethod
    def _name(command: dict[str, Any]) -> str:
        name = command.get("name")
        _require(isinstance(name, str) and bool(name), "item/building name required")
        return name

    @staticmethod
    def _full(actor: Unit, extra: int = 1) -> bool:
        # The request field is authoritative; official character capacities are
        # safe fallbacks for otherwise valid, older payloads omitting that field.
        capacity = actor.capacity if actor.capacity is not None else (100 if actor.kind == "worker" else 40)
        return len(actor.backpack) + extra > capacity

    def _near_zone(self, actor: Unit, kind: str) -> bool:
        return any(zone == kind and distance(actor.pos, pos) == 1
                   for pos, zone in self.turn.zones.items())

    def _build(self, actor: Unit, command: dict[str, Any]) -> tuple[Pos, int, int, int | None]:
        target = self._single_target(command)
        name = self._name(command)
        _require(actor.kind == "worker" and self.turn.is_day, "building requires daytime worker")
        _require(name in (*TOWER_TYPES, "wall"), "unknown building")
        _require(distance(actor.pos, target) == 1, "build target is not adjacent")
        _require(self.turn.land(target), "build target is not land")
        _require(target not in self.reserved_cells, "target already reserved")
        if self._verified_sites is not None:
            _require(target in self._verified_sites.get(name, set()), "unverified construction site")
        replacement = next((u for u in self.turn.weapons() if u.pos == target), None)
        if replacement is not None:
            _require(name in TOWER_TYPES, "cannot replace a weapon with a wall")
            _require(target not in getattr(self.turn, "unknown_occupied", ()),
                     "replacement target has an unparseable occupant")
            # Do not clear other known occupants merely because a weapon is here.
            units = (*self.turn.ours, *getattr(self.turn, "enemies", ()))
            _require(not any(u.health > 0 and u.unit_id != replacement.unit_id
                             and target in self.turn.footprint(u) for u in units),
                     "replacement target has another occupant")
            _require(not any(r.health > 0 and r.pos == target for r in self.turn.robots),
                     "replacement target has a robot")
        else:
            _require(target not in self.turn.blocked(actor), "build target is occupied")
        if name == "wall":
            _require("stone" in actor.backpack, "wall requires one stone")
            return target, 0, 0, None
        delta = 0 if replacement is not None else 1
        _require(self._tower_count + delta <= 3, "three-weapon limit reached")
        return target, WEAPON_BUILD_COST, delta, replacement.unit_id if replacement else None

    def _attack(self, tower: Unit, command: dict[str, Any]) -> int:
        _require(tower.kind in TOWER_TYPES, "only weapons attack")
        _require(not self.turn.is_day, "attacking is only allowed at night")
        _require(tower.cooldown == 0, "weapon is cooling down")
        _require(tower.unit_id not in self._changed_buildings, "weapon is being changed")
        _require(1 <= tower.level <= 3, "invalid weapon level")
        controller_raw = command.get("controllerId")
        _require(isinstance(controller_raw, str) and controller_raw.isascii()
                 and controller_raw.isdecimal(), "controller ID must be a decimal string")
        controller_id = int(controller_raw)
        controller = self._units.get(controller_id)
        _require(controller is not None and controller.health > 0
                 and controller.kind in CONTROLLABLE_TYPES, "unknown or dead controller")
        _require(controller_id not in self.used_actors, "controller already has an action")
        _require(distance(controller.pos, tower.pos) == 1, "controller is not adjacent")
        count = 1 if tower.kind == "railgun" else tower.level
        targets = self._targets(command, count)
        _require(all(0 < distance(tower.pos, p) <= tower.range_of_attack() for p in targets),
                 "attack target is outside range")
        if tower.kind == "gatling":
            vectors = [(p.x - tower.pos.x, p.y - tower.pos.y) for p in targets]
            _require(all(ax * bx + ay * by >= 0 for ax, ay in vectors for bx, by in vectors),
                     "gatling targets exceed 90 degrees")
        return controller_id

    def _trade(self, actor: Unit, command: dict[str, Any], *, buying: bool) -> int:
        name = self._name(command)
        num = command.get("num", 1)
        _require(type(num) is int and num > 0, "quantity must be a positive integer")
        zone = "weaponShop" if buying else "vendor"
        _require(self._near_zone(actor, zone), "trader is not adjacent")
        prices = getattr(self.turn, "shop_prices" if buying else "vendor_prices", {})
        price = prices.get(name)
        _require(type(price) is int and price >= 0, "item price is unavailable")
        if buying:
            _require(not self._full(actor, num), "purchase exceeds backpack capacity")
            return price * num
        _require(name in MINERALS, "only minerals can be sold")
        _require(Counter(actor.backpack)[name] >= num, "not enough minerals to sell")
        return 0

    def _use(self, actor: Unit, command: dict[str, Any]) -> int | None:
        name = self._name(command)
        _require(name in actor.backpack, "item not in backpack")
        if name == "Medicine":
            _require("targetPos" not in command, "Medicine is used by its carrier")
            return None
        if name in ("DizzyWeapon", "Bomb"):
            self._single_target(command)  # Officially no distance restriction.
            return None
        _require(name in UPGRADES or name == "WallFixer",
                 "unsupported consumable (daily summon tracking is disabled)")
        target = self._single_target(command)
        building = next((u for u in self.turn.ours if u.health > 0
                         and u.kind in (*TOWER_TYPES, "wall", "station")
                         and target in self.turn.footprint(u)), None)
        _require(building is not None, "target is not a living friendly building")
        _require(min(distance(actor.pos, p) for p in self.turn.footprint(building)) == 1,
                 "building is not adjacent")
        if name == "WallFixer":
            _require(building.kind == "wall", "WallFixer requires a wall")
        else:
            kinds, level = UPGRADES[name]
            _require(building.kind in kinds and building.level == level,
                     "upgrade voucher does not match building type/level")
        return building.unit_id
