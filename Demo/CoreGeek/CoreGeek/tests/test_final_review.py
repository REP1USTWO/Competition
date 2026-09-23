"""Focused regressions found during final Champion selection (no new search)."""
import copy
import unittest

from test_brain import STATION, payload, role, tower, robot
from agent import brain
from agent.grid import next_step, path_length
from agent.protocol import Pos, Turn


class FinalReviewTests(unittest.TestCase):
    def setUp(self):
        brain.reset_memory()
        self.config = brain.get_config()

    def tearDown(self):
        brain.set_config(self.config)
        brain.reset_memory()

    def test_frozen_champion_delays_third_tower_until_day_two(self):
        brain.set_config(brain.champion_config())
        units = [STATION, tower(10030, 'railgun', 9, 24),
                 tower(10031, 'railgun', 10, 25), role(10010, x=8, y=24)]
        day_one = brain.decide(payload(units, gold=25))
        self.assertFalse(any(c['action'] == 'build' for c in day_one.values()))
        day_two = brain.decide(payload(units, round_no=131, gold=25))
        self.assertEqual(day_two['10010']['action'], 'build')
        self.assertEqual(day_two['10010']['name'], 'rocket')

    def test_match_runner_restores_production_configuration(self):
        from experiments.sim import Scenario, run_match
        selected = brain.champion_config()
        brain.set_config(selected)
        run_match(brain.StrategyConfig(), Scenario(days=0))
        self.assertIs(brain.get_config(), selected)

    def test_astar_already_arrived_returns_no_move(self):
        turn = Turn.load(payload([STATION, role(10010)]))
        worker = turn.workers()[0]
        self.assertEqual(path_length(turn, worker, worker.pos), 0)
        self.assertIsNone(next_step(turn, worker, worker.pos))

    def test_enclosed_character_has_no_path_and_no_crash(self):
        units = [STATION, role(10010, x=2, y=2)]
        units += [tower(40000+i, 'wall', x, y) for i, (x,y) in enumerate(
            ( (x,y) for x in (1,2,3) for y in (1,2,3) if (x,y) != (2,2)))]
        turn = Turn.load(payload(units, gold=0))
        worker = turn.workers()[0]
        self.assertIsNone(next_step(turn, worker, Pos(9,23)))
        self.assertIsNone(path_length(turn, worker, Pos(9,23)))

    def test_astar_does_not_enter_visible_enemy(self):
        request = payload([STATION, role(10010, x=2, y=2)])
        request['teamEnemy']['roles'] = [role(20010, x=3, y=2)]
        turn = Turn.load(request)
        self.assertIsNone(next_step(turn, turn.workers()[0], Pos(3,2)))

    def test_two_adjacent_mines_of_same_type_do_not_compare_pos(self):
        turn = Turn.load(payload([STATION, role(10010, x=7, y=26)], gold=0,
            zones=[{'neutralType': 'iron', 'pos': {'x': x, 'y': 27}} for x in (6, 8)]))
        state = brain._State(turn, brain.Memory())
        state.mem.focus_mineral[10010] = 'iron'
        commands = {}
        brain._mine(state, turn.workers()[0], commands)
        self.assertEqual(commands[10010]['action'], 'collect')

    def test_only_available_post_occupied_by_controller_still_attacks(self):
        cells = [(x, y) for x in (8, 9, 10) for y in (23, 24, 25)
                 if (x, y) not in ((9, 24), (9, 23), (10, 23), (10, 24))]
        units = [STATION, tower(10030, 'railgun', 9, 24), role(10010, x=9, y=23)]
        units.extend(tower(40000+i, 'wall', x, y) for i, (x,y) in enumerate(cells))
        out = brain.decide(payload(units, round_no=71, robots=[robot(30001, x=7, y=22)]))
        self.assertEqual(out.get('10030', {}).get('action'), 'attack')

    def test_dusk_character_at_its_post_does_not_leave_to_work(self):
        turn = Turn.load(payload([STATION, role(10010, x=9, y=23)], round_no=70, gold=0))
        state = brain._State(turn, brain.Memory())
        state.role_post[10010] = Pos(9, 23)
        self.assertIn(10010, brain._returning_roles(state))

    def test_survival_gate_prevents_expected_damage_from_enabling_steal(self):
        units = [STATION, tower(10040, 'rocket', 9, 25, level=3), role(10010, x=8, y=24)]
        targets = [robot(30001, x=9, y=21, targetTeam='challenger'),
                   robot(30002, x=15, y=24, targetTeam='defender')]
        turn = Turn.load(payload(units, round_no=71, robots=targets))
        state = brain._State(turn, brain.Memory())
        chosen = brain._pick_targets(state, state.towers[10040], {30001: 40})
        self.assertEqual(chosen, [Pos(9, 21)] * 3)

    def test_idle_global_rocket_can_target_enemy_robots(self):
        units = [STATION, tower(10040, 'rocket', 9, 25, level=3), role(10010, x=8, y=24)]
        out = brain.decide(payload(units, round_no=71,
            robots=[robot(30002, x=30, y=10, targetTeam='defender')]))
        self.assertEqual(out['10040']['targetPos'], [{'x': 30, 'y': 10}] * 3)

    def test_duplicate_turn_does_not_reset_feedback_backoff(self):
        request = payload([STATION, role(10010, x=9, y=23)], gold=75)
        brain.decide(request)
        brain._MEMORY.build_failures[(Pos(9,24), 'railgun')] = 1
        first = copy.deepcopy(brain._MEMORY.last_commands)
        brain.decide(copy.deepcopy(request))
        self.assertEqual(brain._MEMORY.build_failures.get((Pos(9,24), 'railgun')), 1)
        self.assertEqual(brain._MEMORY.last_commands, first)


if __name__ == '__main__':
    unittest.main()
