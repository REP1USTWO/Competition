"""Evidence-backed construction candidates, deliberately separate from game rules."""

import json
import logging
import os
from pathlib import Path

from .protocol import Pos, Turn, TOWER_TYPES


class Layout:
    def __init__(self, data: dict | None = None) -> None:
        if data is None:
            path = Path(os.environ.get("AGENT_LAYOUT", Path(__file__).resolve().parents[2] / "layout.json"))
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                logging.getLogger(__name__).exception("layout unavailable; only observed building sites are usable")
                data = {}
        if not isinstance(data, dict):
            data = {}
        loadout = data.get("loadout", ["railgun", "railgun", "rocket"])
        self.loadout = tuple(loadout) if isinstance(loadout, list) and len(loadout) == 3 and all(x in TOWER_TYPES for x in loadout) else ("railgun", "railgun", "rocket")
        self.build_walls = data.get("build_walls") is True
        self.bases = data.get("bases", []) if isinstance(data.get("bases"), list) else []

    def sites(self, turn: Turn, kind: str) -> tuple[Pos, ...]:
        station = turn.station()
        if station is None:
            return ()
        result = []
        for base in self.bases:
            try:
                if (base.get("team") != turn.team_type or Pos.load(base["station"]) != station.pos
                        or base.get("width") != turn.width or base.get("height") != turn.height):
                    continue
                for raw in base.get(kind, []):
                    pos = Pos.load(raw)
                    if turn.land(pos) and pos not in result:
                        result.append(pos)
            except (AttributeError, KeyError, ValueError, TypeError):
                logging.getLogger(__name__).warning("ignoring malformed layout entry")
        return tuple(result)
