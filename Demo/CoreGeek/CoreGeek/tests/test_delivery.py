"""Start the real delivery entry from an independent directory and use HTTP."""
import concurrent.futures
import copy
import http.client
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from test_brain import STATION, payload, role, tower, robot
from agent.protocol import Turn
from agent.validator import TurnPlanningContext


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='competition-entry-')
        target = Path(cls.directory.name)
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source / 'src', target / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('main3.py', 'layout.json', 'run.sh'):
            shutil.copy2(source / name, target / name)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.proc = subprocess.Popen([sys.executable, '-B', str(target/'main3.py'), str(cls.port)],
            cwd=target, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(('127.0.0.1', cls.port), timeout=.1):
                    break
            except OSError:
                if cls.proc.poll() is not None or time.monotonic() > deadline:
                    cls.proc.terminate()
                    cls.proc.wait(timeout=3)
                    cls.directory.cleanup()
                    raise RuntimeError('delivery entry did not start')
                time.sleep(.02)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=3)
        cls.directory.cleanup()

    def post(self, body):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=4)
        started = time.perf_counter()
        try:
            connection.request('POST', '/', body, {'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            result = json.loads(response.read())
            self.assertEqual(set(result), {'roleCommandMap', 'prompt', 'executeCmd'})
            return result, time.perf_counter() - started
        finally:
            connection.close()

    @staticmethod
    def request(round_no):
        return payload([STATION, tower(10030, 'railgun', 9, 24), role(10010, x=9, y=23)],
                       round_no=round_no, gold=0, robots=[robot(30001, x=7, y=23)])

    def test_real_entry_25_requests_and_malformed_recovery(self):
        bad, _ = self.post(b'{broken')
        self.assertEqual(bad['roleCommandMap'], {})
        durations = []
        for round_no in range(71, 96):
            request = self.request(round_no)
            result, duration = self.post(json.dumps(request).encode())
            self.assertTrue(result['roleCommandMap'])
            context = TurnPlanningContext(Turn.load(request))
            self.assertEqual(context.validate(result['roleCommandMap']), result['roleCommandMap'])
            durations.append(duration)
        print(f'\n[entry-http] requests=25 max_ms={max(durations)*1000:.2f}')
        self.assertLess(max(durations), 4)

    def test_real_entry_8_concurrent_duplicate_requests(self):
        body = json.dumps(self.request(85)).encode()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: self.post(body)[0], range(8)))
        self.assertTrue(results[0]['roleCommandMap'])
        self.assertTrue(all(result == results[0] for result in results))

    def test_opposite_side_coordinates_remain_legal(self):
        request = copy.deepcopy(self.request(71))
        request['teamOur']['type'] = 'defender'
        request['teamOur']['teamId'] = 'delivery-defender'
        # Reflect occupied station cells: left-top anchor needs a width offset.
        for unit in request['teamOur']['roles']:
            pos = unit['pos']
            if unit['roleType'] == 'station':
                unit['pos'] = {'x': 39-pos['x'], 'y': 32-pos['y']}
            else:
                unit['pos'] = {'x': 40-pos['x'], 'y': 31-pos['y']}
        request['robot']['roles'][0]['pos'] = {'x': 33, 'y': 8}
        request['mapInfo']['zones'] = []
        out, _ = self.post(json.dumps(request).encode())
        self.assertEqual(out['roleCommandMap']['10030']['action'], 'attack')
        context = TurnPlanningContext(Turn.load(request))
        self.assertEqual(context.validate(out['roleCommandMap']), out['roleCommandMap'])
