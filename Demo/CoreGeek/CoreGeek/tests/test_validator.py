"""Regression tests for reservations and official action constraints."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.protocol import Pos, Turn
from agent.validator import TurnPlanningContext


def role(unit_id, kind="worker", x=4, y=4, **extra):
    result = {"id": unit_id, "roleType": kind, "pos": {"x": x, "y": y},
              "health": 220, "level": 1, "cooldown": 0, "backPackCapability": 100,
              "backpack": []}
    result.update(extra)
    return result


def payload(roles, *, gold=75, round_no=1, zones=(), enemies=()):
    return {
        "roundNo": round_no,
        "mapInfo": {"width": 41, "height": 32, "zones": list(zones)},
        "teamOur": {"type": "challenger", "roles": roles, "goldNum": gold},
        "teamEnemy": {"roles": list(enemies)}, "robot": {"roles": []},
        "vendorShopList": [{"name": "stone", "price": 2}, {"name": "iron", "price": 5}],
        "weaponShopList": [{"name": "Medicine", "price": 10},
                           {"name": "WeaponUpgradeVoucher1", "price": 100}],
    }


def context(roles, **kwargs):
    return TurnPlanningContext(Turn.load(payload(roles, **kwargs)))


def at(action, x, y, **extra):
    return {"action": action, "targetPos": [{"x": x, "y": y}], **extra}


def zone(kind, x, y):
    return {"neutralType": kind, "pos": {"x": x, "y": y}}


class ValidatorTests(unittest.TestCase):
    def test_shared_budget_reserves_only_successful_builds(self):
        ctx = context([role(1), role(2, x=6)], gold=25)
        self.assertTrue(ctx.try_add(1, at("build", 4, 5, name="railgun")))
        before = (deepcopy(ctx.commands), set(ctx.reserved_cells), ctx.remaining_gold)
        self.assertFalse(ctx.try_add(2, at("build", 6, 5, name="rocket")))
        self.assertEqual(before, (ctx.commands, ctx.reserved_cells, ctx.remaining_gold))
        self.assertTrue(ctx.try_add(2, at("move", 6, 5)))

    def test_three_tower_limit_includes_planned_towers(self):
        towers = [role(10, "railgun", 10, 10), role(11, "rocket", 12, 10)]
        ctx = context([role(1), role(2, x=6), *towers], gold=100)
        self.assertTrue(ctx.try_add(1, at("build", 4, 5, name="gatling")))
        self.assertFalse(ctx.try_add(2, at("build", 6, 5, name="railgun")))

    def test_existing_three_towers_allow_replacement_only(self):
        towers = [role(10, "railgun", 4, 5), role(11, "rocket", 12, 10),
                  role(12, "gatling", 14, 10)]
        ctx = context([role(1), *towers])
        self.assertFalse(ctx.try_add(1, at("build", 5, 5, name="railgun")))
        self.assertTrue(ctx.try_add(1, at("build", 4, 5, name="rocket")))
        self.assertEqual(ctx.remaining_gold, 50)

    def test_dead_actors_and_pioneer_building_are_rejected(self):
        ctx = context([role(1, health=0), role(2, "pioneer", 6, 4)])
        self.assertFalse(ctx.try_add(1, at("move", 4, 5)))
        self.assertFalse(ctx.try_add(2, at("build", 6, 5, name="railgun")))
        self.assertFalse(ctx.try_add(2, at("collect", 6, 5)))

    def test_build_is_day_only_but_collect_has_no_invented_day_rule(self):
        ctx = context([role(1)], round_no=71, zones=[zone("iron", 4, 5)])
        self.assertFalse(ctx.try_add(1, at("build", 5, 5, name="railgun")))
        self.assertTrue(ctx.try_add(1, at("collect", 4, 5)))

    def test_same_actor_and_destination_cannot_be_reused(self):
        ctx = context([role(1), role(2, x=6)])
        self.assertTrue(ctx.try_add(1, at("move", 5, 5)))
        self.assertFalse(ctx.try_add(1, at("move", 3, 5)))
        self.assertFalse(ctx.try_add(2, at("build", 5, 5, name="railgun")))
        self.assertFalse(ctx.try_add(2, at("move", 5, 5)))

    def test_move_rejects_obstacles_and_non_adjacent_cells(self):
        ctx = context([role(1), role(2, x=5)], zones=[zone("stone", 4, 5)])
        for x, y in ((5, 4), (4, 5), (4, 4), (6, 6), (-1, 4), (41, 4)):
            with self.subTest(target=(x, y)):
                self.assertFalse(ctx.try_add(1, at("move", x, y)))
        self.assertTrue(ctx.try_add(1, at("move", 3, 3)))

    def test_enemy_obstacle(self):
        ctx = context([role(1)], enemies=[role(99, x=5)])
        self.assertFalse(ctx.try_add(1, at("move", 5, 4)))

    def test_full_bag_rejected_and_same_mine_can_be_shared(self):
        mine = [zone("stone", 5, 4)]
        full = context([role(1, backpack=["stone"], backPackCapability=1)], zones=mine)
        self.assertFalse(full.try_add(1, at("collect", 5, 4)))
        ctx = context([role(1), role(2, x=6)], zones=mine)
        self.assertTrue(ctx.try_add(1, at("collect", 5, 4)))
        self.assertTrue(ctx.try_add(2, at("collect", 5, 4)))
        self.assertEqual(ctx.reserved_cells, set())

    def test_wall_requires_material_and_uses_actor_action(self):
        ctx = context([role(1), role(2, x=6, backpack=["stone"])])
        self.assertFalse(ctx.try_add(1, at("build", 4, 5, name="wall")))
        self.assertTrue(ctx.try_add(2, at("build", 6, 5, name="wall")))
        self.assertFalse(ctx.try_add(2, at("build", 7, 5, name="wall")))
        self.assertEqual(ctx.remaining_gold, 75)

    def test_attack_controller_conflicts_in_both_orders(self):
        roles = [role(1), role(10, "railgun", 4, 5), role(11, "rocket", 5, 5)]
        ctx = context(roles, round_no=71)
        self.assertTrue(ctx.try_add(10, at("attack", 7, 7, controllerId="1")))
        self.assertFalse(ctx.try_add(11, at("attack", 7, 7, controllerId="1")))
        self.assertFalse(ctx.try_add(1, at("move", 3, 4)))
        ctx = context(roles, round_no=71)
        self.assertTrue(ctx.try_add(1, at("move", 3, 4)))
        self.assertFalse(ctx.try_add(10, at("attack", 7, 7, controllerId="1")))

    def test_attack_checks_day_cooldown_range_controller(self):
        for changes, round_no, target, controller in (
            ({}, 1, (7, 7), "1"), ({"cooldown": 1}, 71, (7, 7), "1"),
            ({"attackRange": 2}, 71, (10, 10), "1"),
            ({}, 71, (7, 7), "999"), ({}, 71, (7, 7), 1),
        ):
            with self.subTest(changes=changes, round_no=round_no, controller=controller):
                ctx = context([role(1), role(10, "railgun", 4, 5, **changes)], round_no=round_no)
                self.assertFalse(ctx.try_add(10, at("attack", *target, controllerId=controller)))

    def test_gatling_level_count_and_ninety_degree_cone(self):
        ctx = context([role(1), role(10, "gatling", 4, 5, level=3)], round_no=71)
        self.assertFalse(ctx.try_add(10, at("attack", 6, 5, controllerId="1")))
        command = at("attack", 6, 5, controllerId="1")
        command["targetPos"] += [{"x": 4, "y": 7}, {"x": 3, "y": 5}]
        self.assertFalse(ctx.try_add(10, command))
        command["targetPos"][-1] = {"x": 6, "y": 7}
        self.assertTrue(ctx.try_add(10, command))

    def test_rocket_repeated_landing_is_legal_and_railgun_one_target(self):
        ctx = context([role(1), role(10, "rocket", 4, 5, level=3)], round_no=71)
        command = at("attack", 7, 7, controllerId="1")
        command["targetPos"] *= 3
        self.assertTrue(ctx.try_add(10, command))
        ctx = context([role(1), role(10, "railgun", 4, 5, level=3)], round_no=71)
        self.assertTrue(ctx.try_add(10, at("attack", 7, 7, controllerId="1")))

    def test_malformed_commands_fail_closed_and_leave_reservations_unchanged(self):
        bad_commands = [None, [], {}, {"action": []}, {"action": "invalid"},
                        at("move", True, 5), at("move", "4", 5),
                        {"action": "move", "targetPos": "4,5"},
                        {"action": "move", "targetPos": [{}]},
                        at("build", 4, 5, name=[]), at("move", 4, 5, surprise=True)]
        ctx = context([role(1)])
        for command in bad_commands:
            with self.subTest(command=command):
                self.assertFalse(ctx.try_add(1, command))
                self.assertEqual((ctx.commands, ctx.reserved_cells, ctx.remaining_gold), ({}, set(), 75))
        self.assertFalse(ctx.try_add(True, at("move", 4, 5)))
        self.assertTrue(ctx.try_add(1, at("move", 4, 5)))

    def test_buy_checks_budget_capacity_and_price(self):
        ctx = context([role(1), role(2, x=6)], gold=10, zones=[zone("weaponShop", 5, 4)])
        self.assertFalse(ctx.try_add(1, {"action": "buy", "name": "Medicine", "num": 0}))
        self.assertFalse(ctx.try_add(1, {"action": "buy", "name": "Unknown"}))
        self.assertTrue(ctx.try_add(1, {"action": "buy", "name": "Medicine"}))
        self.assertFalse(ctx.try_add(2, {"action": "buy", "name": "Medicine"}))
        full = context([role(1, backPackCapability=0)], zones=[zone("weaponShop", 5, 4)])
        self.assertFalse(full.try_add(1, {"action": "buy", "name": "Medicine"}))

    def test_sell_inventory_and_no_spending_unsettled_income(self):
        ctx = context([role(1, backpack=["stone", "stone"]), role(2, x=6)], gold=0,
                      zones=[zone("vendor", 4, 5), zone("weaponShop", 6, 5)])
        self.assertFalse(ctx.try_add(1, {"action": "sell", "name": "stone", "num": 3}))
        self.assertTrue(ctx.try_add(1, {"action": "sell", "name": "stone", "num": 2}))
        self.assertEqual(ctx.remaining_gold, 0)
        self.assertFalse(ctx.try_add(2, {"action": "buy", "name": "Medicine"}))

    def test_upgrade_matches_kind_level_and_reserves_building(self):
        ctx = context([role(1, backpack=["WeaponUpgradeVoucher1", "WeaponUpgradeVoucher2"]),
                       role(2, x=6, backpack=["WeaponUpgradeVoucher1"]),
                       role(10, "railgun", 5, 5)])
        self.assertFalse(ctx.try_add(1, at("use", 5, 5, name="WeaponUpgradeVoucher2")))
        self.assertTrue(ctx.try_add(1, at("use", 5, 5, name="WeaponUpgradeVoucher1")))
        self.assertFalse(ctx.try_add(2, at("use", 5, 5, name="WeaponUpgradeVoucher1")))

    def test_attack_and_upgrade_cannot_target_same_tower_in_either_order(self):
        roles = [role(1), role(2, x=6, backpack=["WeaponUpgradeVoucher1"]),
                 role(10, "railgun", 5, 5)]
        attack = at("attack", 7, 7, controllerId="1")
        upgrade = at("use", 5, 5, name="WeaponUpgradeVoucher1")
        for first, second in (((10, attack), (2, upgrade)), ((2, upgrade), (10, attack))):
            ctx = context(roles, round_no=71)
            self.assertTrue(ctx.try_add(*first))
            self.assertFalse(ctx.try_add(*second))

    def test_station_upgrade_uses_two_by_two_footprint(self):
        ctx = context([role(1, x=6, y=2, backpack=["StationUpgradeVoucher1"]),
                       role(10, "station", 4, 4)])
        self.assertTrue(ctx.try_add(1, at("use", 4, 4, name="StationUpgradeVoucher1")))

    def test_remove_wall_cannot_consume_it_twice(self):
        ctx = context([role(1), role(2, x=6), role(10, "wall", 5, 5)])
        self.assertTrue(ctx.try_add(1, at("remove", 5, 5)))
        self.assertFalse(ctx.try_add(2, at("remove", 5, 5)))

    def test_cannot_upgrade_wrong_building_or_repair_a_weapon(self):
        ctx = context([role(1, backpack=["WallUpgradeVoucher1", "WallFixer"]),
                       role(10, "railgun", 4, 5)])
        self.assertFalse(ctx.try_add(1, at("use", 4, 5, name="WallUpgradeVoucher1")))
        self.assertFalse(ctx.try_add(1, at("use", 4, 5, name="WallFixer")))

    def test_medicine_and_ranged_items(self):
        ctx = context([role(1, backpack=["Medicine"]), role(2, x=6, backpack=["Bomb"])])
        self.assertFalse(ctx.try_add(1, at("use", 4, 5, name="Medicine")))
        self.assertTrue(ctx.try_add(1, {"action": "use", "name": "Medicine"}))
        self.assertTrue(ctx.try_add(2, at("use", 40, 31, name="Bomb")))

    def test_validate_and_dump_defend_against_caller_mutation(self):
        ctx = context([role(1)])
        cmd = at("move", 4, 5)
        result = ctx.validate({"1": cmd})
        cmd["targetPos"][0]["x"] = 999
        result["1"]["targetPos"][0]["x"] = 888
        self.assertEqual(ctx.commands[1]["targetPos"][0]["x"], 4)

    def test_invalid_ids_and_non_mapping_input_fail_closed(self):
        ctx = context([role(1)])
        self.assertEqual(ctx.validate([]), {})
        self.assertEqual(ctx.validate({"9" * 5000: at("move", 4, 5)}), {})
        self.assertEqual(ctx.validate({"bad": at("move", 4, 5)}), {})
        self.assertEqual(ctx.validate({"1": at("move", 4, 5)})["1"]["action"], "move")

    def test_task_and_summon_policy_is_disabled_explicitly(self):
        ctx = context([role(1, "pioneer", backpack=["SmallRobotSummonOrder"])])
        self.assertFalse(ctx.try_add(1, {"action": "acceptTask"}))
        self.assertIn("disabled", ctx.rejections[-1]["reason"])
        self.assertFalse(ctx.try_add(1, {"action": "use", "name": "SmallRobotSummonOrder"}))
        self.assertIn("disabled", ctx.rejections[-1]["reason"])

    def test_optional_verified_sites_do_not_invent_geometry(self):
        turn = Turn.load(payload([role(1)]))
        ctx = TurnPlanningContext(turn, verified_sites={"railgun": {Pos(4, 5)}})
        self.assertFalse(ctx.try_add(1, at("build", 5, 5, name="railgun")))
        self.assertTrue(ctx.try_add(1, at("build", 4, 5, name="railgun")))


if __name__ == "__main__":
    unittest.main()
