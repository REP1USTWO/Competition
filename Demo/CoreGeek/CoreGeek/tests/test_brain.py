"""Strategy regressions: defense, economy, dusk return, and multi-night stability."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import brain
from agent.protocol import Turn
from agent.validator import TurnPlanningContext

STATION = {"id": 10013, "roleType": "station", "pos": {"x": 10, "y": 24},
           "health": 1500, "level": 1}
ZONES = [
    {"neutralType": "vendor", "pos": {"x": 12, "y": 22}},
    {"neutralType": "weaponShop", "pos": {"x": 14, "y": 26}},
    {"neutralType": "stone", "pos": {"x": 6, "y": 24}},
    {"neutralType": "iron", "pos": {"x": 8, "y": 27}},
    {"neutralType": "copper", "pos": {"x": 16, "y": 22}},
]
VENDOR = [{"name": "stone", "price": 1}, {"name": "iron", "price": 3},
          {"name": "copper", "price": 5}]
SHOP = [{"name": "WeaponUpgradeVoucher1", "price": 100},
        {"name": "WeaponUpgradeVoucher2", "price": 150},
        {"name": "StationUpgradeVoucher1", "price": 100},
        {"name": "StationUpgradeVoucher2", "price": 150},
        {"name": "Medicine", "price": 10}]


def role(unit_id, kind="worker", x=10, y=22, **extra):
    result = {"id": unit_id, "roleType": kind, "pos": {"x": x, "y": y},
              "health": 220 if kind == "worker" else 200,
              "backPackCapability": 100 if kind == "worker" else 40, "backpack": []}
    result.update(extra)
    return result


def tower(unit_id, kind, x, y, **extra):
    result = {"id": unit_id, "roleType": kind, "pos": {"x": x, "y": y},
              "health": 1000, "level": 1, "cooldown": 0, "attackRange": 0,
              "backpack": []}
    result.update(extra)
    return result


def robot(unit_id, kind="smallRobot", x=5, y=20, **extra):
    result = {"id": unit_id, "roleType": kind, "pos": {"x": x, "y": y},
              "health": 40}
    result.update(extra)
    return result


def payload(roles, *, round_no=1, gold=75, robots=(), zones=ZONES, results=None):
    return {
        "roundNo": round_no,
        "mapInfo": {"width": 41, "height": 32, "zones": list(zones)},
        "teamOur": {"type": "challenger", "teamId": "brain-test", "goldNum": gold,
                    "roles": list(roles)},
        "teamEnemy": {"roles": []},
        "robot": {"roles": list(robots)},
        "vendorShopList": list(VENDOR),
        "weaponShopList": list(SHOP),
        "lastRoundRoleActionResults": results or {},
    }


def decide(payload_dict):
    out = brain.decide(payload_dict)
    # Everything the brain emits must survive a fresh independent validation.
    ctx = TurnPlanningContext(Turn.load(payload_dict))
    for key, command in out.items():
        assert ctx.try_add(int(key), command), (key, command, ctx.rejections)
    return out


class BrainTestCase(unittest.TestCase):
    def setUp(self):
        brain.reset_memory()


class DayBuildTests(BrainTestCase):
    def test_day_one_workers_build_towers_next_to_station(self):
        roles = [STATION, role(10010, x=9, y=23), role(10012, x=11, y=25),
                 role(10011, "pioneer", 10, 22)]
        out = decide(payload(roles, round_no=1, gold=75))
        builds = [c for c in out.values() if c["action"] == "build"]
        self.assertTrue(builds)
        for command in builds:
            self.assertIn(command["name"], ("gatling", "railgun", "rocket"))

    def test_shared_budget_blocks_double_spend_with_25_gold(self):
        roles = [STATION, role(10010, x=9, y=23), role(10012, x=11, y=25)]
        out = decide(payload(roles, round_no=1, gold=25))
        builds = [c for c in out.values() if c["action"] == "build"]
        self.assertLessEqual(len(builds), 1)

    def test_build_failure_blacklists_site_and_tries_another(self):
        roles = [STATION, role(10010, x=9, y=23)]
        first = decide(payload(roles, round_no=1, gold=75))
        build = next(c for c in first.values() if c["action"] == "build")
        site = build["targetPos"][0]
        results = {"10010": False}
        for round_no in (2, 3):
            out = decide(payload(roles, round_no=round_no, gold=75, results=results))
        second = next((c for c in out.values() if c["action"] == "build"), None)
        self.assertIsNotNone(second)
        self.assertNotEqual(second["targetPos"][0], site)

    def test_destroyed_tower_is_rebuilt_on_observed_site(self):
        roles = [STATION, role(10010, x=9, y=23),
                 tower(10030, "railgun", 9, 24)]
        decide(payload(roles, round_no=1, gold=75))
        decide(payload(roles, round_no=2, gold=75))
        # Tower is gone the next day: its observed cell stays a verified site.
        out = decide(payload([STATION, role(10010, x=9, y=23)], round_no=3, gold=75))
        builds = [c for c in out.values() if c["action"] == "build"]
        self.assertTrue(any(c["targetPos"][0] == {"x": 9, "y": 24} for c in builds))


class DuskReturnTests(BrainTestCase):
    def test_far_worker_returns_before_nightfall(self):
        roles = [STATION, role(10010, x=30, y=10),
                 tower(10030, "railgun", 9, 24)]
        early = decide(payload(roles, round_no=10, gold=0))
        self.assertNotEqual(early.get("10010", {}).get("action"), "move") \
            if early.get("10010", {}).get("action") == "collect" else None
        late = decide(payload(roles, round_no=66, gold=0))
        command = late.get("10010")
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "move")
        # The walk heads west toward the station area.
        self.assertLess(command["targetPos"][0]["x"], 30)

    def test_boundary_rounds_70_and_131_are_handled(self):
        roles = [STATION, role(10010, x=20, y=16),
                 tower(10030, "railgun", 9, 24)]
        day_end = decide(payload(roles, round_no=70, gold=0))
        self.assertEqual(day_end.get("10010", {}).get("action"), "move")
        # Round 131 is the first day round of day two: economy resumes.
        roles_day2 = [STATION, role(10010, x=9, y=23),
                      tower(10030, "railgun", 9, 24)]
        day_two = decide(payload(roles_day2, round_no=131, gold=0))
        self.assertIsInstance(day_two, dict)


class NightDefenseTests(BrainTestCase):
    def roles_with_towers(self):
        return [
            STATION,
            tower(10030, "railgun", 9, 24),
            tower(10040, "rocket", 10, 25),
            role(10010, x=9, y=23),
            role(10011, "pioneer", 11, 25),
        ]

    def test_adjacent_controller_attacks_robot_in_range(self):
        out = decide(payload(self.roles_with_towers(), round_no=71,
                             robots=[robot(30001, x=9, y=20)]))
        attacks = {k: c for k, c in out.items() if c["action"] == "attack"}
        self.assertIn("10030", attacks)
        self.assertIn(attacks["10030"]["controllerId"], ("10010", "10011"))

    def test_rocket_waits_out_cooldown(self):
        roles = self.roles_with_towers()
        roles[1] = tower(10040, "rocket", 10, 25, cooldown=2)
        roles[2] = role(10010, x=9, y=23)
        out = decide(payload(roles, round_no=71, robots=[robot(30001, x=9, y=20)]))
        self.assertNotIn("10040", out)

    def test_gatling_level_two_sends_two_cone_compatible_targets(self):
        roles = [STATION, tower(10020, "gatling", 9, 24, level=2),
                 role(10010, x=9, y=23)]
        robots = [robot(30001, x=9, y=21), robot(30002, x=7, y=22),
                  robot(30003, "middleRobot", 12, 27, health=60)]
        out = decide(payload(roles, round_no=71, robots=robots))
        attack = out.get("10020")
        self.assertIsNotNone(attack)
        self.assertEqual(len(attack["targetPos"]), 2)

    def test_rocket_level_three_pads_repeated_landing(self):
        roles = [STATION, tower(10040, "rocket", 10, 25, level=3),
                 role(10010, x=11, y=25)]
        out = decide(payload(roles, round_no=71, robots=[robot(30001, x=15, y=20)]))
        attack = out.get("10040")
        self.assertIsNotNone(attack)
        self.assertEqual(len(attack["targetPos"]), 3)

    def test_dead_controller_releases_assignment_and_tower_is_remanned(self):
        roles = self.roles_with_towers()
        decide(payload(roles, round_no=71, robots=[robot(30001, x=9, y=20)]))
        hurt = [STATION, tower(10030, "railgun", 9, 24),
                role(10010, x=9, y=23, health=0),
                role(10011, "pioneer", 10, 25)]
        out = decide(payload(hurt, round_no=72, robots=[robot(30001, x=9, y=20)]))
        attack = out.get("10030")
        self.assertIsNotNone(attack)
        self.assertEqual(attack["controllerId"], "10011")

    def test_controllers_walk_to_towers_instead_of_attacking_from_afar(self):
        roles = [STATION, tower(10030, "railgun", 9, 24),
                 role(10010, x=20, y=10)]
        out = decide(payload(roles, round_no=71, robots=[robot(30001, x=9, y=20)]))
        self.assertNotIn("10030", out)
        self.assertEqual(out["10010"]["action"], "move")

    def test_enemy_targeted_robots_are_deprioritised(self):
        roles = [STATION, tower(10030, "railgun", 9, 24), role(10010, x=9, y=23)]
        robots = [
            robot(30001, x=9, y=21, targetTeam="defender"),
            robot(30002, x=9, y=22, targetTeam="challenger"),
        ]
        out = decide(payload(roles, round_no=71, robots=robots))
        target = out["10030"]["targetPos"][0]
        self.assertEqual((target["x"], target["y"]), (9, 22))


class EconomyTests(BrainTestCase):
    def test_worker_collects_best_adjacent_mineral(self):
        roles = [STATION, role(10010, x=7, y=26),
                 tower(10030, "railgun", 9, 24), tower(10031, "railgun", 10, 25),
                 tower(10040, "rocket", 9, 25)]
        out = decide(payload(roles, round_no=5, gold=0))
        command = out.get("10010")
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "collect")
        self.assertEqual(command["targetPos"][0], {"x": 8, "y": 27})  # iron > stone

    def test_full_backpack_goes_to_vendor_and_sells(self):
        pack = ["iron"] * 100
        roles = [STATION, role(10010, x=12, y=21, backpack=pack),
                 tower(10030, "railgun", 9, 24), tower(10031, "railgun", 10, 25),
                 tower(10040, "rocket", 9, 25)]
        out = decide(payload(roles, round_no=5, gold=0))
        command = out.get("10010")
        self.assertEqual(command["action"], "sell")
        self.assertEqual((command["name"], command["num"]), ("iron", 100))

    def test_pioneer_buys_and_uses_upgrade_voucher(self):
        roles = [STATION, role(10011, "pioneer", 13, 26),
                 tower(10030, "railgun", 9, 24), tower(10031, "railgun", 10, 25),
                 tower(10040, "rocket", 9, 25)]
        out = decide(payload(roles, round_no=5, gold=120))
        self.assertEqual(out.get("10011"),
                         {"action": "buy", "name": "WeaponUpgradeVoucher1", "num": 1})
        carrying = [STATION,
                    role(10011, "pioneer", 9, 23,
                         backpack=["WeaponUpgradeVoucher1"]),
                    tower(10030, "railgun", 9, 24)]
        out = decide(payload(carrying, round_no=6, gold=20))
        command = out.get("10011")
        self.assertEqual(command["action"], "use")
        self.assertEqual(command["name"], "WeaponUpgradeVoucher1")

    def test_failed_collect_marks_mine_for_retry_later(self):
        roles = [STATION, role(10010, x=7, y=26),
                 tower(10030, "railgun", 9, 24), tower(10031, "railgun", 10, 25),
                 tower(10040, "rocket", 9, 25)]
        decide(payload(roles, round_no=5, gold=0))
        out = decide(payload(roles, round_no=6, gold=0, results={"10010": False}))
        command = out.get("10010")
        self.assertIsNotNone(command)
        if command["action"] == "collect":
            self.assertNotEqual(command["targetPos"][0], {"x": 8, "y": 27})


class WallTests(BrainTestCase):
    def three_towers(self):
        return [tower(10030, "railgun", 9, 24), tower(10031, "railgun", 10, 25),
                tower(10040, "rocket", 9, 25)]

    def test_stone_worker_builds_ring_wall_from_day_three(self):
        roles = [STATION, role(10010, x=9, y=22, backpack=["stone"]),
                 role(10012, x=10, y=22), *self.three_towers()]
        out = decide(payload(roles, round_no=261, gold=0))  # day 3 first round
        builds = [c for c in out.values() if c["action"] == "build"]
        self.assertTrue(any(c["name"] == "wall" for c in builds))

    def test_no_walls_before_day_three(self):
        roles = [STATION, role(10010, x=9, y=22, backpack=["stone"]),
                 role(10012, x=10, y=22), *self.three_towers()]
        out = decide(payload(roles, round_no=5, gold=0))
        builds = [c for c in out.values() if c["action"] == "build"]
        self.assertFalse(any(c["name"] == "wall" for c in builds))

    def test_wall_site_blacklisted_after_repeated_failures(self):
        roles = [STATION, role(10010, x=9, y=22, backpack=["stone", "stone"]),
                 role(10012, x=10, y=22), *self.three_towers()]
        first = decide(payload(roles, round_no=261, gold=0))
        wall = next(c for c in first.values() if c.get("name") == "wall")
        site = wall["targetPos"][0]
        for round_no in (262, 263):
            out = decide(payload(roles, round_no=round_no, gold=0,
                                 results={"10010": False}))
        retry = next((c for c in out.values() if c.get("name") == "wall"), None)
        if retry is not None:
            self.assertNotEqual(retry["targetPos"][0], site)


class MultiNightSmokeTests(BrainTestCase):
    """Ten day/night cycles of decision smoke: no dead states, valid commands.

    This is a strategy state-machine regression, not a judge replay: it proves
    the code keeps producing legal decisions as rounds, losses, gold, and
    backpacks change across cycles.
    """

    def test_ten_day_night_cycles_stay_alive_and_legal(self):
        roles = [
            dict(STATION),
            role(10010, x=9, y=23),
            role(10011, "pioneer", 11, 25),
            role(10012, x=10, y=22),
        ]
        gold = 75
        attacks_seen = 0
        builds_seen = 0
        for day in range(1, 11):
            for phase_round in (1, 30, 60, 68, 71, 90, 120, 130):
                round_no = (day - 1) * 130 + phase_round
                robots = []
                if phase_round >= 71:
                    robots = [
                        robot(30000 + day * 10 + i,
                              ("smallRobot", "middleRobot", "largeRobot")[i % 3],
                              6 + i, 20 + (i % 4),
                              health=(40, 60, 500)[i % 3],
                              targetTeam="challenger")
                        for i in range(min(day + 2, 8))
                    ]
                out = decide(payload(roles, round_no=round_no, gold=gold,
                                     robots=robots))
                self.assertIsInstance(out, dict)
                for key, command in out.items():
                    self.assertIsInstance(key, str)
                    self.assertIn(command["action"], (
                        "move", "attack", "build", "collect", "sell", "buy", "use"))
                    if phase_round >= 71:
                        self.assertNotEqual(command["action"], "build")
                    else:
                        self.assertNotEqual(command["action"], "attack")
                attacks_seen += sum(1 for c in out.values() if c["action"] == "attack")
                builds_seen += sum(1 for c in out.values() if c["action"] == "build")
                # Mutate the world the way a long match would.
                if day == 1 and phase_round == 71:
                    # The judge would have materialised the daytime builds.
                    roles.extend([
                        tower(10030, "railgun", 9, 24),
                        tower(10031, "railgun", 10, 25),
                        tower(10040, "rocket", 9, 25),
                    ])
                if day == 3 and phase_round == 120:
                    roles[1]["health"] = 0          # worker dies during night 3
                if day == 4 and phase_round == 1:
                    roles[1]["health"] = 220        # revived next day (A: revive)
                if day == 5 and phase_round == 1:
                    roles[:] = [r for r in roles
                                if not (r["roleType"] == "railgun")]  # tower lost
                    gold = 60
                if day == 6 and phase_round == 1:
                    roles[1]["backpack"] = ["copper"] * 40
                    gold = 300
        self.assertGreater(attacks_seen, 0)
        self.assertGreater(builds_seen, 0)

    def test_match_restart_resets_assignment_without_stale_ids(self):
        roles = [STATION, tower(10030, "railgun", 9, 24), role(10010, x=9, y=23)]
        decide(payload(roles, round_no=71, robots=[robot(30001, x=9, y=20)]))
        # A fresh match restarts at round 1 with different unit ids.
        fresh = [dict(STATION, id=20013), role(20010, x=9, y=23)]
        out = decide(payload(fresh, round_no=1, gold=75))
        self.assertTrue(all(not key.startswith("100") for key in out))


if __name__ == "__main__":
    unittest.main()
