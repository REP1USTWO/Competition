"""Protocol regressions using official inputs and deliberately damaged states."""

import copy
import json
from pathlib import Path
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from agent.protocol import Pos, Turn, Unit, integer, station_footprint


def role(uid=10010, kind="worker", x=2, y=2, health=220, **extra):
    result = {
        "id": uid, "roleType": kind, "pos": {"x": x, "y": y},
        "health": health,
    }
    result.update(extra)
    return result


def request():
    return {
        "roundNo": 1,
        "mapInfo": {"width": 41, "height": 32, "zones": []},
        "teamOur": {"type": "challenger", "teamId": "test-team", "goldNum": 75,
                    "roles": [role()]},
        "teamEnemy": {"roles": []},
        "robot": {"roles": []},
    }


class ProtocolTests(unittest.TestCase):
    def test_official_request_parses_current_state_and_runtime_ranges(self):
        path = Path(__file__).resolve().parents[4] / "docs" / "request.txt"
        payload = json.loads(path.read_text(encoding="utf-8"))
        turn = Turn.load(payload)
        self.assertEqual((turn.round_no, turn.width, turn.height, turn.gold), (85, 41, 32, 20))
        self.assertFalse(turn.is_day)
        self.assertEqual(turn.team_type, "challenger")
        self.assertEqual(turn.team_id, "6324")
        self.assertEqual({u.unit_id for u in turn.controllable()}, {10010, 10011, 10012})
        self.assertEqual(len(turn.robots), 4)
        self.assertEqual(turn.station().pos, Pos(10, 24))
        self.assertEqual(turn.vendor_prices, {"stone": 1, "iron": 3, "copper": 5})
        self.assertFalse(turn.last_results[10010])
        self.assertEqual(turn.shop_prices["WeaponUpgradeVoucher1"], 100)
        self.assertTrue(turn.world_news["folkLegends"])
        # The request sample differs from the narrative range table: live values win.
        self.assertEqual({u.kind: u.range_of_attack() for u in turn.weapons()},
                         {"gatling": 4, "railgun": 7, "rocket": 2147483647})

    def test_malformed_unit_isolated_but_known_position_stays_blocked(self):
        payload = request()
        payload["teamOur"]["roles"].extend([
            role(10012, x=3, y=2, health="not-an-integer"),
            role(10011, "pioneer", x=4, y=2),
        ])
        turn = Turn.load(payload)
        self.assertEqual({u.unit_id for u in turn.controllable()}, {10010, 10011})
        self.assertIn(Pos(3, 2), turn.unknown_occupied)
        self.assertIn(Pos(3, 2), turn.blocked(turn.workers()[0]))
        self.assertTrue(turn.parse_errors)

    def test_malformed_station_preserves_its_whole_footprint(self):
        payload = request()
        payload["teamEnemy"]["roles"] = [role(20013, "station", 30, 10, health="bad")]
        turn = Turn.load(payload)
        self.assertEqual(turn.enemies, ())
        self.assertTrue(set(station_footprint(Pos(30, 10))).issubset(turn.unknown_occupied))

    def test_duplicate_unit_id_cannot_create_two_controllable_roles(self):
        payload = request()
        payload["teamOur"]["roles"].append(role(10010, x=3, y=3))
        turn = Turn.load(payload)
        self.assertEqual(len(turn.controllable()), 1)
        self.assertIn(Pos(3, 3), turn.occupied_cells())

    def test_coordinates_reject_bool_fraction_missing_and_wrong_types(self):
        for raw in ({"x": True, "y": 2}, {"x": 2, "y": False},
                    {"x": 1.5, "y": 2}, {"x": 2}, {"x": "nan", "y": 2},
                    None, [], "2,2"):
            with self.subTest(raw=raw):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    Pos.load(raw)

    def test_integer_does_not_silently_coerce_booleans_or_floats(self):
        for value in (True, False, 1.0, 2.5, None, [], {}):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    integer(value)
        self.assertEqual(integer("12"), 12)

    def test_out_of_bounds_or_invalid_unit_coordinate_does_not_remove_good_role(self):
        for pos in ({"x": -1, "y": 2}, {"x": 41, "y": 2},
                    {"x": 2, "y": 32}, {"x": True, "y": 2}):
            with self.subTest(pos=pos):
                payload = request()
                bad = role(10012)
                bad["pos"] = pos
                payload["teamOur"]["roles"].append(bad)
                turn = Turn.load(payload)
                self.assertEqual([u.unit_id for u in turn.controllable()], [10010])
                self.assertTrue(turn.parse_errors)

    def test_enemy_station_blocks_two_by_two_and_visible_role_blocks_one_cell(self):
        payload = request()
        payload["teamEnemy"]["roles"] = [role(20013, "station", 30, 10, 1500),
                                           role(20010, x=3, y=2)]
        turn = Turn.load(payload)
        blocked = turn.blocked(turn.workers()[0])
        self.assertTrue({Pos(30, 10), Pos(31, 10), Pos(30, 9), Pos(31, 9)}.issubset(blocked))
        self.assertIn(Pos(3, 2), blocked)
        self.assertNotIn(Pos(29, 10), blocked)
        self.assertNotIn(Pos(2, 2), blocked)

    def test_dead_own_roles_and_buildings_do_not_block_or_get_selected(self):
        payload = request()
        payload["teamOur"]["roles"].extend([
            role(10011, "pioneer", 4, 4, 0),
            role(10012, "worker", 5, 5, -1),
            role(10013, "station", 10, 24, 0),
            role(10020, "gatling", 9, 24, 0),
            role(40000, "wall", 8, 24, 0),
        ])
        turn = Turn.load(payload)
        self.assertEqual([u.unit_id for u in turn.controllable()], [10010])
        self.assertEqual(turn.occupied_cells(), frozenset({Pos(2, 2)}))
        self.assertIsNone(turn.station())
        self.assertEqual(turn.weapons(), ())
        self.assertEqual(turn.walls(), ())

    def test_robots_keep_threat_fields_but_only_alive_robots_block(self):
        payload = request()
        payload["robot"]["roles"] = [
            role(30001, "bossRobot", 5, 5, 800, targetTeam="challenger", abnormalState="dizzy"),
            role(30002, "smallRobot", 6, 5, 0),
        ]
        turn = Turn.load(payload)
        self.assertEqual(turn.robots[0].target_team, "challenger")
        self.assertEqual(turn.robots[0].kind, "bossRobot")
        self.assertEqual(turn.robots[0].abnormal_state, "dizzy")
        blocked = turn.blocked(turn.workers()[0])
        self.assertIn(Pos(5, 5), blocked)
        self.assertNotIn(Pos(6, 5), blocked)

    def test_day_night_boundaries_and_remaining_turns(self):
        cases = [(1, True, 1, 70), (70, True, 1, 1), (71, False, 1, 0),
                 (130, False, 1, 0), (131, True, 2, 70), (200, True, 2, 1),
                 (201, False, 2, 0), (1300, False, 10, 0)]
        for round_no, day, day_no, remaining in cases:
            with self.subTest(round_no=round_no):
                payload = request()
                payload["roundNo"] = round_no
                turn = Turn.load(payload)
                self.assertEqual((turn.is_day, turn.day_number, turn.remaining_day_rounds),
                                 (day, day_no, remaining))

    def test_prices_and_boolean_feedback_ignore_damaged_entries(self):
        payload = request()
        payload["vendorShopList"] = [
            {"name": "stone", "price": 0}, {"name": "copper", "price": "5"},
            {"name": "negative", "price": -1}, {"name": "bool", "price": True},
            {"name": "missing"}, None, "bad", {"name": 12, "price": 3},
        ]
        payload["weaponShopList"] = [{"name": "Medicine", "price": 10},
                                     {"name": "Bomb", "price": "bad"}]
        payload["lastRoundRoleActionResults"] = {
            "10010": False, "10011": True, "10012": "false", "10013": 1,
            "invalid": True,
        }
        turn = Turn.load(payload)
        self.assertEqual(turn.vendor_prices, {"stone": 0, "copper": 5})
        self.assertEqual(turn.shop_prices, {"Medicine": 10})
        self.assertEqual(turn.last_results, {10010: False, 10011: True})
        self.assertTrue(turn.parse_errors)

    def test_empty_and_null_arrays_have_safe_defaults(self):
        for empty in (None, [], "wrong-type", {}):
            with self.subTest(empty=empty):
                payload = request()
                payload["mapInfo"]["zones"] = empty
                payload["teamOur"]["roles"] = empty
                payload["teamOur"]["playerTasks"] = empty
                payload["teamEnemy"]["roles"] = empty
                payload["robot"]["roles"] = empty
                payload["vendorShopList"] = empty
                payload["weaponShopList"] = empty
                payload["errors"] = empty
                turn = Turn.load(payload)
                self.assertEqual(turn.controllable(), ())
                self.assertEqual(turn.enemies, ())
                self.assertEqual(turn.robots, ())
                self.assertEqual(turn.zones, {})
                self.assertEqual(turn.vendor_prices, {})
                self.assertEqual(turn.shop_prices, {})
                self.assertEqual(turn.player_tasks, ())
                self.assertEqual(turn.errors, ())

    def test_optional_null_objects_and_nonstring_messages_are_safe(self):
        payload = request()
        payload.update(teamEnemy=None, robot=None, worldNews=None, phaseTask=[],
                       llmResp={}, lastCmdResult=7, lastRoundRoleActionResults=None)
        turn = Turn.load(payload)
        self.assertEqual((turn.enemies, turn.robots, turn.world_news), ((), (), {}))
        self.assertEqual((turn.phase_task, turn.llm_resp, turn.last_cmd_result), ("", "", ""))
        self.assertEqual(turn.last_results, {})

    def test_task_news_and_command_feedback_preserved_without_entering_strategy(self):
        payload = request()
        task = {"taskType": "自进化类1", "timeoutRounds": 20, "isValid": True}
        payload["teamOur"]["playerTasks"] = [task, None]
        payload.update(phaseTask="任务原文", llmResp="model output",
                       lastCmdResult="[TIMEOUT]\npartial", errors=[{"errorCode": 1}, "bad"],
                       worldNews={"officialNews": "新闻", "folkLegends": "传闻", "bad": []})
        turn = Turn.load(payload)
        self.assertEqual(turn.player_tasks, (task,))
        self.assertEqual(turn.phase_task, "任务原文")
        self.assertEqual(turn.llm_resp, "model output")
        self.assertEqual(turn.last_cmd_result, "[TIMEOUT]\npartial")
        self.assertEqual(turn.world_news, {"officialNews": "新闻", "folkLegends": "传闻"})
        self.assertEqual(turn.errors, ({"errorCode": 1},))

    def test_capacity_defaults_and_explicit_full_backpacks(self):
        for kind, default in (("worker", 100), ("pioneer", 40)):
            with self.subTest(kind=kind):
                unit = Unit.load(role(kind=kind, backpack=["stone"] * default))
                self.assertEqual(unit.capacity, default)
                self.assertTrue(unit.backpack_full)
                unit = Unit.load(role(kind=kind, backPackCapability=None))
                self.assertEqual(unit.capacity, default)
                self.assertFalse(unit.backpack_full)
        self.assertTrue(Unit.load(role(backPackCapability=0)).backpack_full)
        self.assertIsNone(Unit.load(role(kind="station")).capacity)

    def test_invalid_backpack_isolated_without_losing_other_roles(self):
        payload = request()
        payload["teamOur"]["roles"].append(role(10012, x=3, y=3, backpack=["stone", 123]))
        turn = Turn.load(payload)
        self.assertEqual([u.unit_id for u in turn.controllable()], [10010])
        self.assertIn(Pos(3, 3), turn.occupied_cells())

    def test_invalid_top_level_and_required_turn_fields_rejected(self):
        invalid = [None, [], "request", 1, {}, {"roundNo": 1},
                   dict(request(), roundNo=True), dict(request(), roundNo=0),
                   dict(request(), roundNo="bad"), dict(request(), mapInfo=None),
                   dict(request(), teamOur=None)]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    Turn.load(payload)

    def test_invalid_dimensions_rejected_before_path_planning(self):
        for width, height in ((0, 32), (41, 0), (-1, 32), (257, 32),
                              (41, 257), (True, 32), (41, 2.5)):
            with self.subTest(width=width, height=height):
                payload = request()
                payload["mapInfo"].update(width=width, height=height)
                with self.assertRaises((ValueError, TypeError)):
                    Turn.load(payload)

    def test_loading_does_not_mutate_request(self):
        payload = request()
        payload["teamOur"]["roles"][0]["backpack"] = ["stone"]
        before = copy.deepcopy(payload)
        Turn.load(payload)
        self.assertEqual(payload, before)


if __name__ == "__main__":
    unittest.main()
