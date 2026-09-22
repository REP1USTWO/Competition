from heapq import heappop, heappush
from itertools import count
from typing import Iterable

from .protocol import Pos, Turn, Unit, distance

_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _search(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    extra_blocked: Iterable[Pos] = (),
) -> tuple[dict[Pos, Pos], int] | None:
    """A* over the 8-direction grid; returns (came_from, cost) or None."""
    if moving.pos == goal:
        return {}, 0
    blocked = set(turn.blocked(moving))
    blocked.update(extra_blocked)
    blocked.discard(moving.pos)
    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [
        (distance(moving.pos, goal), 0, next(order), moving.pos)
    ]
    came_from: dict[Pos, Pos] = {}
    best = {moving.pos: 0}
    seen: set[Pos] = set()

    while frontier:
        _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        if current == goal:
            return came_from, cost
        seen.add(current)
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(
                frontier,
                (
                    new_cost + distance(step, goal),
                    new_cost,
                    next(order),
                    step,
                ),
            )
    return None


def next_step(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    extra_blocked: Iterable[Pos] = (),
) -> Pos | None:
    result = _search(turn, moving, goal, extra_blocked)
    if result is None:
        return None
    came_from, _ = result
    return _first_step(came_from, moving.pos, goal)


def path_length(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    extra_blocked: Iterable[Pos] = (),
) -> int | None:
    """Shortest walkable distance in rounds, or None when unreachable."""
    result = _search(turn, moving, goal, extra_blocked)
    return None if result is None else result[1]


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current
