"""Experiment runner: compare strategy configs over shared seed sets.

Usage: python -m experiments.runner <experiment> [--games N] [--seed-start S]
Writes logs/<experiment>_<tag>/summary.json + games.jsonl + failures.jsonl.
Model reads summaries; raw per-game data stays on disk.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent import brain
from experiments.sim import Scenario, run_match

CFG = brain.StrategyConfig

# Each entry: (variant name, config, scenario overrides)
EXPERIMENTS = {
    "p0-loadout": [
        ("champion-2rail-1rocket", CFG(), {}),
        ("rocket-x3", CFG(loadout=("rocket", "rocket", "rocket")), {}),
        ("rocket-x2-rail", CFG(loadout=("rocket", "rocket", "railgun")), {}),
    ],
    "p2-build-tempo": [
        ("rush-3", CFG(), {}),
        ("min-defense-2", CFG(day1_tower_cap=2), {}),
    ],
    "p3-walls": [
        ("walls-none", CFG(wall_mode="none"), {}),
        ("walls-ring-d3", CFG(), {}),
        ("walls-ring-d2", CFG(wall_start_day=2), {}),
    ],
    "p4-upgrade-delay": [
        ("immediate", CFG(), {}),
        ("hold-80", CFG(upgrade_use_threshold=0.8), {}),
        ("hold-65", CFG(upgrade_use_threshold=0.65), {}),
        ("hold-50", CFG(upgrade_use_threshold=0.5), {}),
    ],
    "p6-sell-timing": [
        ("sell-full", CFG(), {}),
        ("sell-85", CFG(sell_fill_fraction=0.85), {}),
        ("sell-70", CFG(sell_fill_fraction=0.70), {}),
        ("sell-50", CFG(sell_fill_fraction=0.50), {}),
    ],
    "p7-gold-reserve": [
        ("reserve-0", CFG(), {}),
        ("reserve-25", CFG(gold_reserve=25), {}),
        ("reserve-50", CFG(gold_reserve=50), {}),
    ],
    "p11-consumables": [
        ("champion", CFG(), {}),
        ("dizzy-d6", CFG(consumables_enabled=True), {}),
        ("dizzy-d4", CFG(consumables_enabled=True, consumable_from_day=4), {}),
        ("bomb-d6", CFG(consumables_enabled=True, consumable_name="Bomb"), {}),
    ],
    "p12-wall-workers": [
        ("walls-1-worker", CFG(), {}),
        ("walls-2-workers", CFG(wall_workers=2), {}),
        ("walls-none", CFG(wall_mode="none"), {}),
    ],
    "p13-upgrade-order": [
        ("station-l3-first", CFG(), {}),
        ("weapons-l3-first", CFG(station_l3_first=False), {}),
    ],
}

METRIC_KEYS = [
    "survival_days", "score_total", "score_combat", "station_hp", "kills",
    "gold_earned", "upgrades", "illegal", "attack_misses", "walls_built",
    "weapon_idle_rounds", "weapon_unmanned_rounds", "rocket_shots",
    "worker_idle_rounds", "time_to_3_towers", "time_to_first_upgrade",
    "station_damage",
]


def aggregate(records: list[dict]) -> dict:
    summary = {}
    for key in METRIC_KEYS:
        values = [r[key] for r in records]
        summary[key] = {
            "mean": round(statistics.fmean(values), 2),
            "min": min(values),
            "p10": round(sorted(values)[max(0, len(values) // 10 - 1)]
                         if len(values) >= 10 else min(values), 2),
        }
    summary["games"] = len(records)
    summary["destroyed"] = sum(1 for r in records if r["destroyed_round"])
    return summary


def run_experiment(name: str, games: int, seed_start: int, out_root: Path,
                   scenario_overrides: dict | None = None) -> dict:
    variants = EXPERIMENTS[name]
    result = {"experiment": name, "games": games, "seed_start": seed_start,
              "variants": {}}
    for variant, config, overrides in variants:
        merged = {**(scenario_overrides or {}), **overrides}
        records = []
        for offset in range(games):
            scenario = Scenario(seed=seed_start + offset, **merged)
            metrics = run_match(config, scenario)
            records.append(metrics.dump())
        out_dir = out_root / f"{name}_s{seed_start}_g{games}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{variant}.games.jsonl").open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        failures = sorted(records, key=lambda r: (r["survival_days"], r["score_total"]))
        with (out_dir / f"{variant}.failures.jsonl").open("w", encoding="utf-8") as fh:
            for record in failures[: max(1, len(records) // 10)]:
                fh.write(json.dumps(record) + "\n")
        result["variants"][variant] = aggregate(records)
    return result


def print_table(result: dict) -> None:
    keys = ["survival_days", "score_total", "score_combat", "station_hp",
            "kills", "gold_earned", "upgrades", "illegal", "attack_misses",
            "weapon_idle_rounds", "worker_idle_rounds", "time_to_3_towers"]
    header = f"{'variant':<24}" + "".join(f"{k[:14]:>15}" for k in keys) + f"{'destroyed':>10}"
    print(f"\n== {result['experiment']} (games={result['games']}, seeds>={result['seed_start']}) ==")
    print(header)
    for variant, summary in result["variants"].items():
        row = f"{variant:<24}"
        for key in keys:
            row += f"{summary[key]['mean']:>15}"
        row += f"{summary['destroyed']:>10}"
        print(row)
    for variant, summary in result["variants"].items():
        print(f"  worst {variant}: survival_min={summary['survival_days']['min']} "
              f"score_p10={summary['score_total']['p10']} "
              f"hp_min={summary['station_hp']['min']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--wave-mult", type=float, default=None)
    parser.add_argument("--out", default=str(ROOT.parents[1] / "logs"))
    args = parser.parse_args()
    names = list(EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    overrides = {"wave_mult": args.wave_mult} if args.wave_mult else None
    out_root = Path(args.out)
    for name in names:
        result = run_experiment(name, args.games, args.seed_start, out_root, overrides)
        out_dir = out_root / f"{name}_s{args.seed_start}_g{args.games}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print_table(result)


if __name__ == "__main__":
    main()
