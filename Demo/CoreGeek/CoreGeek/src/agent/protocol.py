from dataclasses import dataclass, field
from typing import Any

DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS
WEAPON_BUILD_COST = 25
WALL_MATERIAL = "stone"
LAND = "land"
STATION = "station"
WALL = "wall"
WORKER = "worker"
PIONEER = "pioneer"
TOWER_TYPES = ("gatling", "railgun", "rocket")
CONTROLLABLE_TYPES = (WORKER, PIONEER)
MINERALS = ("stone", "iron", "copper")
TOWER_RANGE_BY_LEVEL = {"gatling": (3, 5, 7), "railgun": (6, 8, 10), "rocket": (10, 15, 10**9)}


def integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("expected integer")
    return int(value)


def objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class Pos:
    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        return cls(integer(raw["x"]), integer(raw["y"]))

    def dump(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


def distance(first: Pos, second: Pos) -> int:
    return max(abs(first.x - second.x), abs(first.y - second.y))


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    return (pos, Pos(pos.x + 1, pos.y), Pos(pos.x, pos.y - 1), Pos(pos.x + 1, pos.y - 1))


@dataclass(frozen=True, slots=True)
class Unit:
    unit_id: int
    pos: Pos
    kind: str
    health: int
    level: int
    cooldown: int
    attack_range: int
    capacity: int | None
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        kind = str(raw["roleType"])
        capacity = raw.get("backPackCapability")
        if capacity is None and kind in CONTROLLABLE_TYPES:
            capacity = 100 if kind == WORKER else 40
        backpack = raw.get("backpack") or []
        if not isinstance(backpack, list) or any(not isinstance(item, str) for item in backpack):
            raise ValueError("invalid backpack")
        uid = integer(raw["id"])
        if uid <= 0:
            raise ValueError("invalid unit id")
        return cls(uid, Pos.load(raw["pos"]), kind, integer(raw["health"]),
                   integer(raw.get("level") or 1), max(0, integer(raw.get("cooldown") or 0)),
                   max(0, integer(raw.get("attackRange") or 0)),
                   max(0, integer(capacity)) if capacity is not None else None, tuple(backpack))

    @property
    def backpack_full(self) -> bool:
        return self.capacity is not None and len(self.backpack) >= self.capacity

    def range_of_attack(self) -> int:
        if self.attack_range > 0:
            return self.attack_range
        table = TOWER_RANGE_BY_LEVEL.get(self.kind)
        return table[min(max(self.level, 1), 3) - 1] if table else 0


@dataclass(frozen=True, slots=True)
class Robot:
    robot_id: int
    pos: Pos
    health: int
    kind: str = "smallRobot"
    target_team: str = ""
    abnormal_state: str = ""

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        return cls(integer(raw["id"]), Pos.load(raw["pos"]), integer(raw["health"]),
                   str(raw.get("roleType", "smallRobot")), str(raw.get("targetTeam", "")),
                   str(raw.get("abnormalState", "")))


@dataclass(frozen=True, slots=True)
class Turn:
    round_no: int
    is_day: bool
    gold: int
    width: int
    height: int
    zones: dict[Pos, str]
    ours: tuple[Unit, ...]
    robots: tuple[Robot, ...]
    enemies: tuple[Unit, ...] = ()
    team_type: str = ""
    team_id: str = ""
    vendor_prices: dict[str, int] = field(default_factory=dict)
    shop_prices: dict[str, int] = field(default_factory=dict)
    last_results: dict[int, bool] = field(default_factory=dict)
    player_tasks: tuple[dict[str, Any], ...] = ()
    phase_task: str = ""
    world_news: dict[str, str] = field(default_factory=dict)
    llm_resp: str = ""
    last_cmd_result: str = ""
    errors: tuple[dict[str, Any], ...] = ()
    parse_errors: tuple[str, ...] = ()
    unknown_occupied: frozenset[Pos] = frozenset()

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        round_no = integer(payload["roundNo"])
        info, team = payload["mapInfo"], payload["teamOur"]
        if not isinstance(info, dict) or not isinstance(team, dict) or round_no < 1:
            raise ValueError("invalid turn")
        width, height = integer(info["width"]), integer(info["height"])
        # Resource guard, not an additional game rule.
        if not 1 <= width <= 256 or not 1 <= height <= 256:
            raise ValueError("unsupported map dimensions")
        issues: list[str] = []
        unknown: set[Pos] = set()

        def inside(pos: Pos) -> bool:
            return 0 <= pos.x < width and 0 <= pos.y < height

        def units(raw: Any, loader: Any) -> tuple:
            loaded, ids = [], set()
            for item in objects(raw):
                try:
                    unit = loader(item)
                    uid = unit.unit_id if isinstance(unit, Unit) else unit.robot_id
                    if uid in ids or not inside(unit.pos):
                        raise ValueError("duplicate id or invalid position")
                    ids.add(uid)
                    loaded.append(unit)
                except (KeyError, TypeError, ValueError, OverflowError):
                    issues.append("invalid unit")
                    try:
                        pos = Pos.load(item["pos"])
                        if inside(pos):
                            unknown.update(station_footprint(pos) if item.get("roleType") == STATION else (pos,))
                    except (KeyError, TypeError, ValueError, OverflowError):
                        pass
            return tuple(loaded)

        zones = {}
        for zone in objects(info.get("zones")):
            try:
                pos = Pos.load(zone["pos"])
                if inside(pos):
                    zones[pos] = str(zone["neutralType"])
            except (KeyError, TypeError, ValueError, OverflowError):
                issues.append("invalid zone")

        def prices(raw: Any) -> dict[str, int]:
            result = {}
            for item in objects(raw):
                try:
                    price = integer(item["price"])
                    if isinstance(item["name"], str) and price >= 0:
                        result[item["name"]] = price
                except (KeyError, TypeError, ValueError, OverflowError):
                    issues.append("invalid price")
            return result

        results = {}
        raw_results = payload.get("lastRoundRoleActionResults")
        if isinstance(raw_results, dict):
            for key, value in raw_results.items():
                try:
                    if isinstance(value, bool):
                        results[integer(key)] = value
                except (TypeError, ValueError):
                    pass
        enemy, robot, news = payload.get("teamEnemy"), payload.get("robot"), payload.get("worldNews")
        ours = units(team.get("roles"), Unit.load)
        enemies = units(enemy.get("roles") if isinstance(enemy, dict) else None, Unit.load)
        robots = units(robot.get("roles") if isinstance(robot, dict) else None, Robot.load)
        vendor, shop = prices(payload.get("vendorShopList")), prices(payload.get("weaponShopList"))
        return cls(round_no, (round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
                   max(0, integer(team.get("goldNum") or 0)), width, height, zones, ours, robots, enemies,
                   str(team.get("type", "")), str(team.get("teamId", "")), vendor, shop, results,
                   tuple(objects(team.get("playerTasks"))),
                   payload.get("phaseTask") if isinstance(payload.get("phaseTask"), str) else "",
                   {k: v for k, v in news.items() if isinstance(v, str)} if isinstance(news, dict) else {},
                   payload.get("llmResp") if isinstance(payload.get("llmResp"), str) else "",
                   payload.get("lastCmdResult") if isinstance(payload.get("lastCmdResult"), str) else "",
                   tuple(objects(payload.get("errors"))), tuple(issues), frozenset(unknown))

    @property
    def day_number(self) -> int:
        return (self.round_no - 1) // ROUNDS_PER_DAY + 1

    @property
    def remaining_day_rounds(self) -> int:
        return DAY_ROUNDS - (self.round_no - 1) % ROUNDS_PER_DAY if self.is_day else 0

    def station(self) -> Unit | None:
        return next((u for u in self.ours if u.kind == STATION and u.health > 0), None)

    def alive(self, kinds: tuple[str, ...]) -> tuple[Unit, ...]:
        return tuple(u for u in self.ours if u.kind in kinds and u.health > 0)

    def controllable(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive(CONTROLLABLE_TYPES), key=lambda u: u.unit_id))

    def workers(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive((WORKER,)), key=lambda u: u.unit_id))

    def weapons(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive(TOWER_TYPES), key=lambda u: (u.pos.x, u.pos.y)))

    def walls(self) -> tuple[Unit, ...]:
        return self.alive((WALL,))

    def stone_mines(self) -> tuple[Pos, ...]:
        return tuple(p for p, kind in self.zones.items() if kind == WALL_MATERIAL)

    def footprint(self, unit: Unit) -> tuple[Pos, ...]:
        return station_footprint(unit.pos) if unit.kind == STATION else (unit.pos,)

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def land(self, pos: Pos) -> bool:
        return self.in_bounds(pos) and self.zones.get(pos, LAND) == LAND

    def occupied_cells(self) -> frozenset[Pos]:
        cells: set[Pos] = set(self.unknown_occupied)
        for unit in (*self.ours, *self.enemies):
            if unit.health > 0:
                cells.update(self.footprint(unit))
        cells.update(robot.pos for robot in self.robots if robot.health > 0)
        return frozenset(cells)

    def blocked(self, moving: Unit) -> frozenset[Pos]:
        cells = {p for p, kind in self.zones.items() if kind != LAND}
        cells.update(self.occupied_cells())
        cells.discard(moving.pos)
        return frozenset(cells)


def move_command(pos: Pos) -> dict[str, Any]:
    return {"action": "move", "targetPos": [pos.dump()]}


def collect_command(pos: Pos) -> dict[str, Any]:
    return {"action": "collect", "targetPos": [pos.dump()]}


def build_command(pos: Pos, name: str) -> dict[str, Any]:
    return {"action": "build", "targetPos": [pos.dump()], "name": name}


def attack_command(controller_id: int, pos: Pos | list[Pos]) -> dict[str, Any]:
    targets = [pos] if isinstance(pos, Pos) else pos
    return {"action": "attack", "targetPos": [p.dump() for p in targets], "controllerId": str(controller_id)}
