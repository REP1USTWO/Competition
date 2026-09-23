"""Parameterized match simulator for strategy experiments.

This is a simplified mini-judge for A/B strategy comparison, NOT the official
judger. Wave sizes, robot AI, and spawn patterns are SYNTHETIC ASSUMPTIONS;
results compare strategies relative to each other, never absolute win rates.
"""

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import brain
from agent.protocol import Pos, distance, station_footprint

WIDTH, HEIGHT = 41, 32
DAY_ROUNDS, CYCLE = 70, 130
STATION_POS = (10, 24)
ROBOTS = {
    "smallRobot": {"hp": 40, "atk": 5, "range": 3, "score": 1},
    "middleRobot": {"hp": 60, "atk": 10, "range": 3, "score": 2},
    "largeRobot": {"hp": 500, "atk": 20, "range": 3, "score": 4},
    "bossRobot": {"hp": 800, "atk": 40, "range": 3, "score": 10},
}
TOWER_HP = {1: 1000, 2: 1500, 3: 2000}
TOWER_RANGE = {"gatling": (3, 5, 7), "railgun": (6, 8, 10), "rocket": (10, 15, 10**9)}
DEFAULT_MINES = {"stone": [(6, 24), (14, 3)], "iron": [(8, 27), (25, 10)],
                 "copper": [(16, 22), (7, 2)]}
VENDOR = (12, 22)
SHOP = (14, 26)
PRICES = {"stone": 1, "iron": 3, "copper": 5}
SHOP_PRICES = {"WeaponUpgradeVoucher1": 100, "WeaponUpgradeVoucher2": 150,
               "StationUpgradeVoucher1": 100, "StationUpgradeVoucher2": 150,
               "Medicine": 10, "DizzyWeapon": 100, "Bomb": 100}


@dataclass
class Scenario:
    seed: int = 7
    days: int = 10
    wave_base: int = 3           # SYNTHETIC ASSUMPTION: night 1 wave size
    wave_per_day: int = 2        # SYNTHETIC ASSUMPTION: daily growth
    wave_mult: float = 1.0       # scenario pressure multiplier
    hp_mult: float = 1.0
    large_from_day: int = 4
    boss_from_day: int = 7
    spawn_min_dist: int = 12
    start_gold: int = 75
    mines: dict | None = None    # None -> DEFAULT_MINES
    enemy_target_fraction: float = 0.0  # share of wave targeting the enemy team


@dataclass
class Metrics:
    survival_days: int = 0
    station_hp: int = 0
    station_level: int = 1
    destroyed_round: int = 0
    kills: int = 0
    score_combat: int = 0
    score_survival: int = 0
    score_total: int = 0
    gold_earned: int = 0
    gold_spent: int = 0
    gold_end: int = 0
    upgrades: int = 0
    illegal: int = 0
    attack_misses: int = 0
    walls_built: int = 0
    commands: int = 0
    time_to_3_towers: int = 0
    time_to_first_upgrade: int = 0
    weapon_idle_rounds: int = 0
    weapon_unmanned_rounds: int = 0
    rocket_shots: int = 0
    worker_idle_rounds: int = 0
    worker_move_rounds: int = 0
    worker_collect_rounds: int = 0
    station_damage: int = 0
    station_hp_by_day: list = field(default_factory=list)
    towers_end: list = field(default_factory=list)

    def dump(self) -> dict:
        return dict(self.__dict__)


class World:
    def __init__(self, scenario: Scenario):
        self.sc = scenario
        self.rng = random.Random(scenario.seed)
        self._round_no = 0
        self.gold = scenario.start_gold
        self.station = {"id": 10013, "hp": 1500, "level": 1}
        self.roles = {
            10010: {"kind": "worker", "pos": (9, 23), "hp": 220, "cap": 100,
                    "pack": [], "dead_until": 0},
            10011: {"kind": "pioneer", "pos": (11, 25), "hp": 200, "cap": 40,
                    "pack": [], "dead_until": 0},
            10012: {"kind": "worker", "pos": (10, 22), "hp": 220, "cap": 100,
                    "pack": [], "dead_until": 0},
        }
        self.towers = {}
        self.walls = {}
        self.robots = {}
        mines = scenario.mines or DEFAULT_MINES
        self.mines = {Pos(*p): kind for kind, poss in mines.items() for p in poss}
        self.mine_used = {pos: 0 for pos in self.mines}
        self.results = {}
        self.next_robot_id = 30001
        self.tower_ids = {"gatling": iter((10020, 10021, 10022)),
                          "railgun": iter((10030, 10031, 10032)),
                          "rocket": iter((10040, 10041, 10042))}
        self.m = Metrics()

    # -- geometry ----------------------------------------------------------
    def footprint(self):
        return {(p.x, p.y) for p in station_footprint(Pos(*STATION_POS))}

    def occupied(self):
        cells = set(self.footprint()) if self.station["hp"] > 0 else set()
        cells.update(t["pos"] for t in self.towers.values())
        cells.update(self.walls)
        cells.update(r["pos"] for r in self.roles.values() if r["hp"] > 0)
        cells.update(r["pos"] for r in self.robots.values())
        cells.update(self.mines)
        cells.update((VENDOR, SHOP))
        return cells

    @staticmethod
    def land(pos):
        return 0 <= pos[0] < WIDTH and 0 <= pos[1] < HEIGHT

    # -- payload -----------------------------------------------------------
    def payload(self, round_no):
        roles = []
        if self.station["hp"] > 0:
            roles.append({"id": 10013, "roleType": "station",
                          "pos": {"x": STATION_POS[0], "y": STATION_POS[1]},
                          "health": self.station["hp"], "level": self.station["level"]})
        for uid, r in self.roles.items():
            roles.append({"id": uid, "roleType": r["kind"],
                          "pos": {"x": r["pos"][0], "y": r["pos"][1]},
                          "health": r["hp"], "backPackCapability": r["cap"],
                          "backpack": list(r["pack"])})
        for uid, t in self.towers.items():
            roles.append({"id": uid, "roleType": t["kind"],
                          "pos": {"x": t["pos"][0], "y": t["pos"][1]},
                          "health": t["hp"], "level": t["level"],
                          "cooldown": t["cooldown"]})
        for i, (pos, hp) in enumerate(self.walls.items()):
            roles.append({"id": 40000 + i, "roleType": "wall",
                          "pos": {"x": pos[0], "y": pos[1]}, "health": hp, "level": 1})
        zones = ([{"neutralType": kind, "pos": {"x": p.x, "y": p.y}}
                  for p, kind in self.mines.items()]
                 + [{"neutralType": "vendor", "pos": {"x": VENDOR[0], "y": VENDOR[1]}},
                    {"neutralType": "weaponShop", "pos": {"x": SHOP[0], "y": SHOP[1]}}])
        return {
            "roundNo": round_no,
            "mapInfo": {"width": WIDTH, "height": HEIGHT, "zones": zones},
            "teamOur": {"type": "challenger", "teamId": "sim", "goldNum": self.gold,
                        "roles": roles},
            "teamEnemy": {"roles": []},
            "robot": {"roles": [
                {"id": uid, "pos": {"x": r["pos"][0], "y": r["pos"][1]},
                 "roleType": r["kind"], "health": r["hp"],
                 "targetTeam": r.get("target", "challenger"),
                 "abnormalState": "dizzy" if r.get("dizzy", 0) > 0 else ""}
                for uid, r in self.robots.items()]},
            "vendorShopList": [{"name": n, "price": p} for n, p in PRICES.items()],
            "weaponShopList": [{"name": n, "price": p} for n, p in SHOP_PRICES.items()],
            "lastRoundRoleActionResults": dict(self.results),
        }

    # -- waves ---------------------------------------------------------------
    def spawn_wave(self, day):
        sc = self.sc
        count = round((sc.wave_base + day * sc.wave_per_day) * sc.wave_mult)
        kinds = (["smallRobot"] * 5 + ["middleRobot"] * 3
                 + (["largeRobot"] if day >= sc.large_from_day else [])
                 + (["bossRobot"] if day >= sc.boss_from_day else []))
        for _ in range(count):
            for _attempt in range(60):
                pos = (self.rng.randrange(WIDTH), self.rng.randrange(HEIGHT))
                if (distance(Pos(*pos), Pos(*STATION_POS)) >= sc.spawn_min_dist
                        and pos not in self.occupied() and self.land(pos)):
                    break
            else:
                continue
            kind = self.rng.choice(kinds)
            hp = round(ROBOTS[kind]["hp"] * sc.hp_mult)
            target = ("defender" if self.rng.random() < sc.enemy_target_fraction
                      else "challenger")
            self.robots[self.next_robot_id] = {"kind": kind, "pos": pos, "hp": hp,
                                               "dizzy": 0, "target": target}
            self.next_robot_id += 1

    # -- command application -------------------------------------------------
    def apply(self, round_no, out, is_day):
        self.results = {}
        for key, command in out.items():
            uid = int(key)
            ok = self._apply_one(uid, command, is_day)
            self.results[uid] = ok
            if not ok:
                self.m.illegal += 1
                if command.get("action") == "attack":
                    self.m.attack_misses += 1
            elif command.get("action") == "attack" and self.towers.get(uid, {}).get("kind") == "rocket":
                self.m.rocket_shots += 1
        for tower in self.towers.values():
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1

    def _apply_one(self, uid, command, is_day):
        action = command.get("action")
        role = self.roles.get(uid)
        tower = self.towers.get(uid)
        if action == "move" and role and role["hp"] > 0:
            target = (command["targetPos"][0]["x"], command["targetPos"][0]["y"])
            if distance(Pos(*role["pos"]), Pos(*target)) == 1 and self.land(target) \
                    and target not in self.occupied():
                role["pos"] = target
                return True
            return False
        if action == "build" and is_day and role and role["kind"] == "worker":
            name = command["name"]
            site = (command["targetPos"][0]["x"], command["targetPos"][0]["y"])
            if name == "wall":
                if (distance(Pos(*role["pos"]), Pos(*site)) == 1
                        and "stone" in role["pack"] and site not in self.occupied()
                        and self.land(site)):
                    role["pack"].remove("stone")
                    self.walls[site] = 1000
                    self.m.walls_built += 1
                    return True
                return False
            if (distance(Pos(*role["pos"]), Pos(*site)) == 1 and self.gold >= 25
                    and len(self.towers) < 3 and site not in self.occupied()
                    and self.land(site)):
                self.gold -= 25
                self.m.gold_spent += 25
                self.towers[next(self.tower_ids[name])] = {
                    "kind": name, "pos": site, "hp": 1000, "level": 1, "cooldown": 0}
                if len(self.towers) == 3 and not self.m.time_to_3_towers:
                    self.m.time_to_3_towers = self._round_no
                return True
            return False
        if action == "collect" and role and role["kind"] == "worker":
            site = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
            if site in self.mines and distance(Pos(*role["pos"]), site) == 1 \
                    and len(role["pack"]) < role["cap"]:
                role["pack"].append(self.mines[site])
                self.mine_used[site] += 1
                if self.mine_used[site] >= 10:
                    self._respawn_mine(site)
                return True
            return False
        if action == "sell" and role:
            name, num = command["name"], command.get("num", 1)
            if distance(Pos(*role["pos"]), Pos(*VENDOR)) == 1 \
                    and role["pack"].count(name) >= num:
                for _ in range(num):
                    role["pack"].remove(name)
                self.gold += PRICES[name] * num
                self.m.gold_earned += PRICES[name] * num
                return True
            return False
        if action == "buy" and role:
            name = command["name"]
            price = SHOP_PRICES.get(name)
            if price is not None and distance(Pos(*role["pos"]), Pos(*SHOP)) == 1 \
                    and self.gold >= price and len(role["pack"]) < role["cap"]:
                self.gold -= price
                self.m.gold_spent += price
                role["pack"].append(name)
                return True
            return False
        if action == "use" and role:
            return self._use(role, command, round_no_hint=None)
        if action == "attack" and not is_day and tower:
            return self._attack(tower, command)
        return False

    def _use(self, role, command, round_no_hint=None):
        name = command["name"]
        if name not in role["pack"]:
            return False
        if name == "Medicine":
            role["pack"].remove(name)
            role["hp"] = 220 if role["kind"] == "worker" else 200
            return True
        if name in ("DizzyWeapon", "Bomb") and command.get("targetPos"):
            centre = Pos(command["targetPos"][0]["x"], command["targetPos"][0]["y"])
            role["pack"].remove(name)
            for robot in list(self.robots.values()):
                if distance(Pos(*robot["pos"]), centre) <= 1:
                    if name == "Bomb":
                        robot["hp"] -= 100
                    else:
                        robot["dizzy"] = 5
            self._collect_dead()
            return True
        vouchers = {"WeaponUpgradeVoucher1": (1, ("gatling", "railgun", "rocket")),
                    "WeaponUpgradeVoucher2": (2, ("gatling", "railgun", "rocket")),
                    "StationUpgradeVoucher1": (1, ("station",)),
                    "StationUpgradeVoucher2": (2, ("station",))}
        if name not in vouchers or not command.get("targetPos"):
            return False
        level, kinds = vouchers[name]
        site = (command["targetPos"][0]["x"], command["targetPos"][0]["y"])
        if "station" in kinds and site == STATION_POS and self.station["level"] == level:
            role["pack"].remove(name)
            self.station["level"] += 1
            self.station["hp"] = 1500 * self.station["level"]
            self._note_upgrade()
            return True
        for tower in self.towers.values():
            if tower["pos"] == site and tower["kind"] in kinds and tower["level"] == level:
                if distance(Pos(*role["pos"]), Pos(*tower["pos"])) == 1:
                    role["pack"].remove(name)
                    tower["level"] += 1
                    tower["hp"] = TOWER_HP[tower["level"]]
                    self._note_upgrade()
                    return True
        return False

    def _note_upgrade(self):
        self.m.upgrades += 1
        if not self.m.time_to_first_upgrade:
            self.m.time_to_first_upgrade = self._round_no

    def _attack(self, tower, command):
        if tower["cooldown"] > 0:
            return False
        controller = self.roles.get(int(command["controllerId"]))
        if controller is None or controller["hp"] <= 0 \
                or distance(Pos(*controller["pos"]), Pos(*tower["pos"])) != 1:
            return False
        reach = TOWER_RANGE[tower["kind"]][tower["level"] - 1]
        targets = [(p["x"], p["y"]) for p in command["targetPos"]]
        by_pos = {r["pos"]: r for r in self.robots.values()}
        hit = False
        if tower["kind"] == "rocket":
            tower["cooldown"] = 3
            for target in targets:
                for pos, robot in by_pos.items():
                    gap = distance(Pos(*pos), Pos(*target))
                    if gap == 0:
                        robot["hp"] -= 20
                        hit = True
                    elif gap == 1:
                        robot["hp"] -= 10
                        hit = True
        elif tower["kind"] == "gatling":
            for target in targets:
                if target in by_pos and distance(Pos(*tower["pos"]), Pos(*target)) <= reach:
                    by_pos[target]["hp"] -= 10
                    hit = True
        else:
            energy = 10 * tower["level"]
            target = targets[0]
            if distance(Pos(*tower["pos"]), Pos(*target)) <= reach:
                for pos in self._line(tower["pos"], target):
                    robot = by_pos.get(pos)
                    if robot and energy > 0:
                        dealt = min(energy, robot["hp"])
                        robot["hp"] -= dealt
                        energy -= dealt
                        hit = True
        for uid in [u for u, r in self.robots.items() if r["hp"] <= 0]:
            kind = self.robots[uid]["kind"]
            del self.robots[uid]
            self.m.kills += 1
            self.m.score_combat += ROBOTS[kind]["score"]
        return hit

    def _collect_dead(self):
        for uid in [u for u, r in self.robots.items() if r["hp"] <= 0]:
            kind = self.robots[uid]["kind"]
            del self.robots[uid]
            self.m.kills += 1
            self.m.score_combat += ROBOTS[kind]["score"]

    @staticmethod
    def _line(start, end):
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cells = []
        x, y = x0, y0
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            cells.append((x, y))
        return cells

    # -- robot turn -----------------------------------------------------------
    def robots_act(self):
        footprint = self.footprint() if self.station["hp"] > 0 else set()
        tower_pos = {t["pos"] for t in self.towers.values()}
        for robot in list(self.robots.values()):
            if robot.get("dizzy", 0) > 0:
                robot["dizzy"] -= 1
                continue
            spec = ROBOTS[robot["kind"]]
            rpos = Pos(*robot["pos"])
            if robot.get("target", "challenger") != "challenger":
                # Enemy-targeted robots march on the enemy base (30,10); they
                # ignore us entirely in this harness.
                goal = (30, 10)
                dx = (goal[0] > robot["pos"][0]) - (goal[0] < robot["pos"][0])
                dy = (goal[1] > robot["pos"][1]) - (goal[1] < robot["pos"][1])
                nxt = (robot["pos"][0] + dx, robot["pos"][1] + dy)
                if nxt not in self.occupied() and self.land(nxt):
                    robot["pos"] = nxt
                continue
            candidates = (
                [(distance(rpos, Pos(*p)), ("tower", p)) for p in tower_pos]
                + [(distance(rpos, Pos(*p)), ("wall", p)) for p in self.walls]
                + [(distance(rpos, Pos(*c)), ("station", c)) for c in footprint]
                + [(distance(rpos, Pos(*r["pos"])), ("role", uid))
                   for uid, r in self.roles.items() if r["hp"] > 0]
            )
            candidates.sort()
            if candidates and candidates[0][0] <= spec["range"]:
                kind, ref = candidates[0][1]
                if kind == "tower":
                    self.towers[next(u for u, t in self.towers.items()
                                     if t["pos"] == ref)]["hp"] -= spec["atk"]
                elif kind == "wall":
                    self.walls[ref] -= spec["atk"]
                    if self.walls[ref] <= 0:
                        del self.walls[ref]
                elif kind == "station":
                    self.station["hp"] -= spec["atk"]
                    self.m.station_damage += spec["atk"]
                else:
                    self.roles[ref]["hp"] -= spec["atk"]
                continue
            goal = min(footprint or tower_pos or {STATION_POS},
                       key=lambda c: distance(rpos, Pos(*c)))
            dx = (goal[0] > robot["pos"][0]) - (goal[0] < robot["pos"][0])
            dy = (goal[1] > robot["pos"][1]) - (goal[1] < robot["pos"][1])
            nxt = (robot["pos"][0] + dx, robot["pos"][1] + dy)
            if nxt not in self.occupied() and self.land(nxt):
                robot["pos"] = nxt
        for uid in [u for u, t in self.towers.items() if t["hp"] <= 0]:
            del self.towers[uid]

    def _respawn_mine(self, site):
        kind = self.mines.pop(site)
        self.mine_used.pop(site)
        for _ in range(50):
            pos = Pos(self.rng.randrange(WIDTH), self.rng.randrange(HEIGHT))
            if (pos.x, pos.y) not in self.occupied() and self.land((pos.x, pos.y)):
                self.mines[pos] = kind
                self.mine_used[pos] = 0
                return

    def revive_roles(self, round_no):
        for r in self.roles.values():
            if r["hp"] <= 0 and r["dead_until"] and round_no >= r["dead_until"]:
                for cell in sorted(self.footprint()):
                    for cand in ((cell[0] - 1, cell[1]), (cell[0], cell[1] + 1),
                                 (cell[0] + 2, cell[1]), (cell[0], cell[1] - 2)):
                        if cand not in self.occupied() and self.land(cand):
                            r["pos"] = cand
                            r["hp"] = 220 if r["kind"] == "worker" else 200
                            r["dead_until"] = 0
                            break
                    else:
                        continue
                    break

    def mark_deaths(self, round_no):
        for r in self.roles.values():
            if r["hp"] <= 0 and r["dead_until"] == 0:
                r["dead_until"] = (round_no // CYCLE + 1) * CYCLE + 20


def run_match(config: brain.StrategyConfig | None = None,
              scenario: Scenario | None = None) -> Metrics:
    """Drive one full simulated match with the real production strategy."""
    scenario = scenario or Scenario()
    world = World(scenario)
    previous_config = brain.get_config()
    brain.reset_memory()
    brain.set_config(config or brain.champion_config())
    try:
        for round_no in range(1, CYCLE * scenario.days + 1):
            world._round_no = round_no
            is_day = (round_no - 1) % CYCLE < DAY_ROUNDS
            day = (round_no - 1) // CYCLE + 1
            if is_day and (round_no - 1) % CYCLE == 0:
                world.robots.clear()
            if not is_day and (round_no - 1) % CYCLE == DAY_ROUNDS:
                world.spawn_wave(day)
            world.revive_roles(round_no)
            out = brain.decide(world.payload(round_no))
            world.m.commands += len(out)
            _idle_accounting(world, out, is_day)
            world.apply(round_no, out, is_day)
            if not is_day:
                world.robots_act()
            world.mark_deaths(round_no)
            if world.station["hp"] <= 0 and not world.m.destroyed_round:
                world.m.destroyed_round = round_no
                world.m.survival_days = day - 1
            if (round_no % CYCLE) == 0:
                world.m.station_hp_by_day.append(max(0, world.station["hp"]))
            if world.station["hp"] <= 0:
                # Keep responding (the harness must not crash) but stop scoring.
                pass
    finally:
        brain.set_config(previous_config)
        brain.reset_memory()
    m = world.m
    if not m.destroyed_round:
        m.survival_days = scenario.days
    m.station_hp = max(0, world.station["hp"])
    m.station_level = world.station["level"]
    m.gold_end = world.gold
    m.score_survival = sum(10 * d for d in range(1, m.survival_days + 1))
    m.score_total = m.score_combat + m.score_survival
    m.towers_end = [(t["kind"], t["level"], t["hp"]) for t in world.towers.values()]
    return m


def _idle_accounting(world: World, out: dict, is_day: bool) -> None:
    if is_day:
        for uid, r in world.roles.items():
            if r["kind"] != "worker" or r["hp"] <= 0:
                continue
            command = out.get(str(uid))
            if command is None:
                world.m.worker_idle_rounds += 1
            elif command["action"] == "move":
                world.m.worker_move_rounds += 1
            elif command["action"] == "collect":
                world.m.worker_collect_rounds += 1
        return
    for uid, tower in world.towers.items():
        attack = out.get(str(uid))
        if attack is not None:
            continue
        controlled = any(
            r["hp"] > 0 and distance(Pos(*r["pos"]), Pos(*tower["pos"])) == 1
            for r in world.roles.values())
        if controlled and tower["cooldown"] == 0:
            world.m.weapon_idle_rounds += 1
        elif not controlled:
            world.m.weapon_unmanned_rounds += 1
