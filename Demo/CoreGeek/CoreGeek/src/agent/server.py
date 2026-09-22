"""Small, bounded HTTP transport for the competition's turn protocol."""

import json
import logging
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .brain import decide

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 1024 * 1024
BODY_TIMEOUT_SECONDS = 1.0


def empty_response() -> dict[str, Any]:
    return {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}


def normalize_response(result: Any) -> dict[str, Any]:
    """Accept legacy role maps as well as the complete response envelope."""
    if not isinstance(result, dict):
        raise ValueError("decision must return a JSON object")
    if any(key in result for key in ("roleCommandMap", "prompt", "executeCmd")):
        commands = result.get("roleCommandMap", {})
        prompt = result.get("prompt", "")
        execute_cmd = result.get("executeCmd", "")
    else:
        commands, prompt, execute_cmd = result, "", ""
    if not isinstance(commands, dict):
        raise ValueError("roleCommandMap must be a JSON object")
    if not isinstance(prompt, str) or not isinstance(execute_cmd, str):
        raise ValueError("prompt and executeCmd must be strings")
    return {"roleCommandMap": commands, "prompt": prompt, "executeCmd": execute_cmd}


class Handler(BaseHTTPRequestHandler):
    # Can be overridden by a handler subclass, or by server.decider.
    decider = staticmethod(decide)

    def setup(self) -> None:
        self.request.settimeout(getattr(self.server, "body_timeout", BODY_TIMEOUT_SECONDS))
        super().setup()

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionError, socket.timeout, OSError):
            # A departed client is not a decision failure or a server failure.
            self.close_connection = True

    def _read_payload(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Transfer-Encoding is unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise ValueError("exactly one Content-Length is required")
        text_length = lengths[0].strip()
        if not text_length or any(char not in "0123456789" for char in text_length):
            raise ValueError("invalid Content-Length")
        # Bound the integer conversion too (including Python's digit limit).
        maximum = getattr(self.server, "max_body_bytes", MAX_BODY_BYTES)
        if len(text_length) > 12:
            raise ValueError("Content-Length is too large")
        length = int(text_length)
        if not 0 < length <= maximum:
            raise ValueError("request body is empty or too large")
        timeout = getattr(self.server, "body_timeout", BODY_TIMEOUT_SECONDS)
        deadline = time.monotonic() + timeout
        chunks = []
        remaining = length
        while remaining:
            seconds_left = deadline - time.monotonic()
            if seconds_left <= 0:
                raise TimeoutError("request body deadline exceeded")
            self.connection.settimeout(seconds_left)
            chunk = self.rfile.read1(min(65536, remaining))
            if not chunk:
                raise ValueError("incomplete request body")
            chunks.append(chunk)
            remaining -= len(chunk)
        self.connection.settimeout(timeout)
        payload = json.loads(b"".join(chunks).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        return payload

    def do_POST(self) -> None:
        response = empty_response()
        try:
            payload = self._read_payload()
        except (ValueError, UnicodeError, OSError, RecursionError):
            # Do not try to reuse an ambiguous or partially read request stream.
            self.close_connection = True
            LOGGER.warning("invalid request; returning empty commands")
        else:
            try:
                decision = getattr(self.server, "decider", None) or self.decider
                response = normalize_response(decision(payload))
            except Exception:
                LOGGER.exception("decision failed; returning empty commands")
        self._send_response(response)

    def _send_response(self, response: dict[str, Any]) -> None:
        try:
            body = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError):
            LOGGER.exception("invalid decision output; returning empty commands")
            body = json.dumps(empty_response()).encode("utf-8")
        try:
            # The turn protocol expects a JSON response even for a failed turn.
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, socket.timeout, OSError):
            self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(port: int, decider: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
    with ThreadingHTTPServer(("0.0.0.0", port), Handler) as server:
        if decider is not None:
            server.decider = decider
        server.serve_forever()
