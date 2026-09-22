"""Exercise real sockets, including bad input followed by healthy requests."""

import concurrent.futures
import http.client
import json
import logging
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.server import Handler, MAX_BODY_BYTES, empty_response


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.decider = lambda payload: {"1": {"commandType": "move", "x": 1, "y": 2}}
        self.server.body_timeout = 0.15
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.old_log_level = logging.getLogger("agent.server").level
        logging.getLogger("agent.server").setLevel(logging.CRITICAL)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        logging.getLogger("agent.server").setLevel(self.old_log_level)

    def post(self, body=b"{}"):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request("POST", "/", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("application/json", response.getheader("Content-Type"))
            raw = response.read()
            self.assertEqual(len(raw), int(response.getheader("Content-Length")))
            result = json.loads(raw)
            self.assertEqual(set(result), {"roleCommandMap", "prompt", "executeCmd"})
            return result
        finally:
            connection.close()

    def raw_post(self, headers, body=b"", shutdown=False):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
            request = b"POST / HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n" + body
            connection.sendall(request)
            if shutdown:
                connection.shutdown(socket.SHUT_WR)
            response = http.client.HTTPResponse(connection)
            response.begin()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Connection"), "close")
            return json.loads(response.read())

    def test_full_response_and_legacy_response(self):
        self.assertEqual(self.post()["roleCommandMap"]["1"]["commandType"], "move")
        expected = {"roleCommandMap": {}, "prompt": "中文任务", "executeCmd": "pwd"}
        self.server.decider = lambda payload: expected
        self.assertEqual(self.post(), expected)

    def test_twenty_five_continuous_requests(self):
        for round_number in range(25):
            self.assertTrue(self.post(json.dumps({"roundNo": round_number}).encode())["roleCommandMap"])

    def test_eight_concurrent_requests(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: self.post(), range(8)))
        self.assertTrue(all(result["roleCommandMap"] for result in results))

    def test_malformed_requests_do_not_poison_later_requests(self):
        for body in (b"{", b"[]", b"null", b"123", b"\xff", b"", b"[" * 2000 + b"]" * 2000):
            with self.subTest(body=body):
                self.assertEqual(self.post(body), empty_response())
                self.assertTrue(self.post()["roleCommandMap"])

    def test_invalid_lengths_and_transfer_encoding(self):
        cases = [
            b"",
            b"Content-Length: -1\r\n",
            b"Content-Length: bad\r\n",
            b"Content-Length: 0\r\n",
            f"Content-Length: {MAX_BODY_BYTES + 1}\r\n".encode(),
            b"Content-Length: " + b"9" * 100 + b"\r\n",
            b"Content-Length: 2\r\nContent-Length: 2\r\n",
            b"Transfer-Encoding: chunked\r\n",
            b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
        ]
        for headers in cases:
            with self.subTest(headers=headers):
                self.assertEqual(self.raw_post(headers), empty_response())
                self.assertTrue(self.post()["roleCommandMap"])

    def test_short_body_timeout_and_eof(self):
        start = time.monotonic()
        self.assertEqual(self.raw_post(b"Content-Length: 10\r\n", b"{"), empty_response())
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(self.raw_post(b"Content-Length: 10\r\n", b"{", shutdown=True), empty_response())
        self.assertTrue(self.post()["roleCommandMap"])

    def test_empty_and_missing_state_fail_safely(self):
        from agent.brain import decide
        self.server.decider = decide
        self.assertEqual(self.post(), empty_response())
        self.assertEqual(self.post(b'{"roundNo":1}'), empty_response())
        self.server.decider = lambda payload: {}
        self.assertEqual(self.post(), empty_response())

    def test_body_deadline_is_total_not_per_chunk(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
            connection.sendall(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 20\r\n\r\n{")
            stop = threading.Event()
            def trickle():
                while not stop.wait(0.03):
                    try:
                        connection.sendall(b" ")
                    except OSError:
                        break
            sender = threading.Thread(target=trickle, daemon=True)
            sender.start()
            started = time.monotonic()
            try:
                response = http.client.HTTPResponse(connection)
                response.begin()
                self.assertEqual(json.loads(response.read()), empty_response())
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                stop.set()
                sender.join(timeout=1)

    def test_bad_decision_results_fail_safely(self):
        for result in (None, [], {"roleCommandMap": []}, {"prompt": None}, {"executeCmd": []}, {"1": object()}, {"1": float("nan")}):
            with self.subTest(result=result):
                self.server.decider = lambda payload, result=result: result
                self.assertEqual(self.post(), empty_response())
        def broken(payload):
            raise RuntimeError("test failure")
        self.server.decider = broken
        self.assertEqual(self.post(), empty_response())
        self.server.decider = lambda payload: {}
        self.assertEqual(self.post(), empty_response())

    def test_client_disconnect_does_not_stop_server(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
            connection.sendall(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 999\r\n\r\n{")
        self.assertTrue(self.post()["roleCommandMap"])


if __name__ == "__main__":
    unittest.main()
