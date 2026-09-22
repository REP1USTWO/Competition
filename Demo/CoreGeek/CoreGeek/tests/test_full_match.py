"""Full-match simulation: a simplified mini-judge driving 10 day/night cycles.

This is NOT the official judger and cannot prove real-match survival. It is a
long-run stability harness: simplified but rule-shaped world dynamics (A/B
where known, C otherwise) exercise the strategy across 1300 rounds so crashes,
dead states, illegal commands, and broken recovery show up as test failures.
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import brain
from agent.protocol import Pos, Turn, distance, station_footprint
from agent.validator import TurnPlanningContext

WIDTH, HEIGHT = 41, 32
DAY_ROUNDS, CYCLE = 70, 130
STATION_POS = (10, 24)
ROBOTS = {
    "smallRobot": {"hp": 40, "atk": 5, "range": 3},
    "middleRobot": {"hp": 60, "atk": 10, "range": 3},
    "largeRobot": {"hp": 500, "atk": 20, "range": 3},
    "bossRobot": {"hp": 800, "atk": 40, "range": 3},
}
TOWER_HP = {1: 1000, 2: 1500, 3: 2000}
TOWER_RANGE = {"gatling": (3, 5, 7), "railgun": (6, 8, 10), "rocket": (10, 15, 10**9)}
MINES = {"stone": [(6, 24), (14, 3)], "iron": [(8, 27), (25, 10)],
         "copper": [(16, 22), (7, 2)]}
VENDOR = (12, 22)
SHOP = (14, 26)
PRICES = {"stone": 1, "iron": 3, "copper": 5}
SHOP_PRICES = {"WeaponUpgradeVoucher1": 100, "WeaponUpgradeVoucher2": 150,
               "StationUpgradeVoucher1": 100, "StationUpgradeVoucher2": 150,
               "Medicine": 10}


class World:
    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.gold = 75
        self.station = {"id": 10013, "hp": 1500, "level": 1}
        self.roles = {
            10010: {"kind": "worker", "pos": (9, 23), "hp": 220, "cap": 100,
                    "pack": [], "dead_until": 0},
            10011: {"kind": "pioneer", "pos": (11, 25), "hp": 200, "cap": 40,
                    "pack": [], "dead_until": 0},
            10012: {"kind": "worker", "pos": (10, 22), "hp": 220, "cap": 100,
                    "pack": [], "dead_until": 0},
        }
        self.towers = {}        # id -> {kind,pos,hp,level,cooldown}
        self.walls = {}         # pos -> hp
        self.wall_counter = 40000
        self.robots = {}        # id -> {kind,pos,hp}
        self.mines = {Pos(*p): kind for kind, poss in MINES.items() for p in poss}
        self.mine_used = {pos: 0 for pos in self.mines}
        self.results = {}
        self.next_robot_id = 30001
        self.tower_ids = {"gatling": iter((10020, 10021, 10022)),
                          "railgun": iter((10030, 10031, 10032)),
                          "rocket": iter((10040, 10041, 10042))}
        self.kills = 0
        self.upgrades = 0
        self.gold_earned = 0
        self.illegal = 0

    # -- world helpers ----------------------------------------------------
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

    def land(self, pos):
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
                 "targetTeam": "challenger", "abnormalState": ""}
                for uid, r in self.robots.items()]},
            "vendorShopList": [{"name": n, "price": p} for n, p in PRICES.items()],
            "weaponShopList": [{"name": n, "price": p} for n, p in SHOP_PRICES.items()],
            "lastRoundRoleActionResults": dict(self.results),
        }

    # -- night waves --------------------------------------------------------
    def spawn_wave(self, day):
        count = 3 + day * 2
        kinds = (["smallRobot"] * 5 + ["middleRobot"] * 3
                 + (["largeRobot"] if day >= 4 else [])
                 + (["bossRobot"] if day >= 7 else []))
        for _ in range(count):
            while True:
                pos = (self.rng.randrange(WIDTH), self.rng.randrange(HEIGHT))
                far = distance(Pos(*pos), Pos(*STATION_POS)) >= 12
                if far and pos not in self.occupied() and self.land(pos):
                    break
            kind = self.rng.choice(kinds)
            self.robots[self.next_robot_id] = {
                "kind": kind, "pos": pos, "hp": ROBOTS[kind]["hp"]}
            self.next_robot_id += 1

    # -- command application -------------------------------------------------
    def apply(self, round_no, out, is_day):
        self.results = {}
        ctx_turn = Turn.load(self.payload(round_no))
        for key, command in out.items():
            uid = int(key)
            ok = self._apply_one(uid, command, is_day)
            self.results[uid] = ok
            if not ok:
                self.illegal += 1
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
                    return True
                return False
            if (distance(Pos(*role["pos"]), Pos(*site)) == 1 and self.gold >= 25
                    and len(self.towers) < 3 and site not in self.occupied()
                    and self.land(site)):
                self.gold -= 25
                self.towers[next(self.tower_ids[name])] = {
                    "kind": name, "pos": site, "hp": 1000, "level": 1, "cooldown": 0}
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
                self.gold_earned += PRICES[name] * num
                return True
            return False
        if action == "buy" and role:
            name = command["name"]
            price = SHOP_PRICES.get(name)
            if price is not None and distance(Pos(*role["pos"]), Pos(*SHOP)) == 1 \
                    and self.gold >= price and len(role["pack"]) < role["cap"]:
                self.gold -= price
                role["pack"].append(name)
                return True
            return False
        if action == "use" and role:
            return self._use(role, command)
        if action == "attack" and not is_day and tower:
            return self._attack(tower, command)
        return False

    def _use(self, role, command):
        name = command["name"]
        if name not in role["pack"]:
            return False
        if name == "Medicine":
            role["pack"].remove(name)
            role["hp"] = 220 if role["kind"] == "worker" else 200
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
            self.upgrades += 1
            return True
        for tower in self.towers.values():
            if tower["pos"] == site and tower["kind"] in kinds and tower["level"] == level:
                if min(distance(Pos(*role["pos"]), Pos(*c))
                       for c in (tower["pos"],)) == 1:
                    role["pack"].remove(name)
                    tower["level"] += 1
                    tower["hp"] = TOWER_HP[tower["level"]]
                    self.upgrades += 1
                    return True
        return False

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
                for pos, robot in list(by_pos.items()):
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
        else:  # railgun: penetrate along the line with 10*level energy
            energy = 10 * tower["level"]
            target = targets[0]
            if distance(Pos(*tower["pos"]), Pos(*target)) <= reach:
                line = self._line(tower["pos"], target)
                for pos in line:
                    robot = by_pos.get(pos)
                    if robot and energy > 0:
                        dealt = min(energy, robot["hp"])
                        robot["hp"] -= dealt
                        energy -= dealt
                        hit = True
        dead = [uid for uid, r in self.robots.items() if r["hp"] <= 0]
        for uid in dead:
            del self.robots[uid]
            self.kills += 1
        return hit

    @staticmethod
    def _line(start, end):
        # Bresenham between cell centres, matching "相连的直线" (A).
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
        targets = {t["pos"]: t for t in self.towers.values()}
        footprint = self.footprint() if self.station["hp"] > 0 else set()
        for robot in list(self.robots.values()):
            spec = ROBOTS[robot["kind"]]
            rpos = Pos(*robot["pos"])
            # attack the nearest unit inside weapon range
            victim = None
            candidates = (
                [(distance(rpos, Pos(*p)), ("tower", p)) for p in targets]
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
                else:
                    self.roles[ref]["hp"] -= spec["atk"]
                continue
            # otherwise walk toward the station footprint
            goal = min(footprint or targets or {STATION_POS},
                       key=lambda c: distance(rpos, Pos(*c)))
            dx = (goal[0] > robot["pos"][0]) - (goal[0] < robot["pos"][0])
            dy = (goal[1] > robot["pos"][1]) - (goal[1] < robot["pos"][1])
            nxt = (robot["pos"][0] + dx, robot["pos"][1] + dy)
            if nxt not in self.occupied() and self.land(nxt):
                robot["pos"] = nxt
        fallen = [u for u, t in self.towers.items() if t["hp"] <= 0]
        for uid in fallen:
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

    def revive_roles(self, round_no, day_start):
        for r in self.roles.values():
            if r["hp"] <= 0 and round_no >= r["dead_until"]:
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
                # A: revives within the first 20 rounds of the next day.
                r["dead_until"] = (round_no // CYCLE + 1) * CYCLE + 20


class FullMatchTests(unittest.TestCase):
    def setUp(self):
        brain.reset_memory()

    def test_ten_simulated_days_without_crash_or_dead_state(self):
        world = World()
        commands_total = 0
        for round_no in range(1, CYCLE * 10 + 1):
            is_day = (round_no - 1) % CYCLE < DAY_ROUNDS
            day = (round_no - 1) // CYCLE + 1
            if is_day and (round_no - 1) % CYCLE == 0:
                world.robots.clear()        # A: leftovers cleared at dawn
                world.revive_roles(round_no, True)
            if not is_day and (round_no - 1) % CYCLE == DAY_ROUNDS:
                world.spawn_wave(day)
            world.revive_roles(round_no, False)
            out = brain.decide(world.payload(round_no))
            commands_total += len(out)
            self.assertIsInstance(out, dict)
            world.apply(round_no, out, is_day)
            if not is_day:
                world.robots_act()
            world.mark_deaths(round_no)
        # Stability evidence (simulation-scoped, not a real-match guarantee):
        # the loop keeps emitting commands, earns gold, upgrades, kills robots,
        # and the simplified waves do not permanently stall the strategy.
        self.assertGreater(commands_total, 1000)
        self.assertGreater(world.kills, 50)
        self.assertGreater(world.gold_earned, 200)
        self.assertGreater(world.upgrades, 0)
        self.assertLess(world.illegal, commands_total * 0.2)
        # In this simplified harness the base holds for all ten days; this is
        # a regression anchor for the strategy, not a real-match guarantee.
        self.assertGreater(world.station["hp"], 0)
        print(f"\n[sim] station hp={world.station['hp']} level={world.station['level']}"
              f" towers={[(t['kind'], t['level'], t['hp']) for t in world.towers.values()]}"
              f" kills={world.kills} gold_earned={world.gold_earned}"
              f" upgrades={world.upgrades} illegal={world.illegal}"
              f" commands={commands_total}")


if __name__ == "__main__":
    unittest.main()
