# OVERNIGHT REPORT — Strategy Evolution

Branch: `strategy/overnight-evolution`
Baseline (Initial Champion): `1d3e977` on main — loadout (railgun, railgun, rocket),
ring walls from day 3, immediate upgrades, sell-when-full, reserve 25g only while towers missing.

## Method

- Platform: `experiments/sim.py` (parameterized mini-judge; wave sizes/robot AI are
  SYNTHETIC ASSUMPTIONS) + `experiments/runner.py` (seeded A/B, summary.json/games.jsonl/failures.jsonl).
- Strategy knobs: `agent.brain.StrategyConfig`; production strategy code is shared, never forked.
- Train seeds: 1000+. Validation seeds: 2000+. Quick elim 10 games, confirm 30+.
- Score = combat (kills × official points 1/2/4/10) + survival (Σ 10×day). Task score out of scope.

## Hypotheses verdicts (updated as experiments complete)

- 3 Rocket loadout (H1): **REJECTED** — see P0.
- Rocket cooldown/controller rotation (H1b): covered by P0 (rotation did not beat mixed).
- Rush vs economy-first build tempo (H2): pending P2.
- Threat-facing half wall (H3): pending P3.
- Delayed upgrade as heal reserve (H4): pending P4.
- Resource selection (P5): pending.
- Backpack/sell timing (P6): pending.
- Gold reserve (P7): pending.
- Worker specialization (P8): pending.
- Safe cross-map scoring (P9): not started.
- Balanced aggressiveness (P10): not started.

## Experiments

### P0-loadout — 3 Rocket vs Champion vs Rocket×2+Railgun (30 games, seeds 1000–1029)

| variant | survival | score | combat | hp_end | kills | destroyed |
|---|---|---|---|---|---|---|
| champion 2rail+1rocket | 9.93 | **804.7** | 261.0 | **3239** | 131.0 | **1** |
| rocket×3 | 9.60 | 760.1 | 246.1 | 2065 | 128.0 | 5 |
| rocket×2+rail | 9.73 | 771.4 | 245.7 | 2464 | 127.5 | 3 |

Worst case: champion survival_min=8, score_p10=779; rocket×3 survival_min=6, score_p10=439.

Decision: **REJECT 3-rocket, KEEP champion loadout.** Rockets alone lose sustained DPS
(cooldown 3) and worst-case robustness; railgun penetration carries mid waves.
Failure anatomy (shared bad seed): days 1–6 clean, boss waves day 7–9 overwhelm DPS,
towers die, unmanned rounds spike → station falls day 8–9. Bottleneck is late-night
DPS + tower survival, not economy.

### P2-build-tempo — Rush 3 towers vs min-defense-2 (30 games, seeds 1000–1029)

| variant | survival | score | hp_end | destroyed | time_to_3_towers |
|---|---|---|---|---|---|
| rush-3 (champion) | 9.93 | **804.7** | 3239 | 1 | 6 rounds |
| min-defense-2 | 10.0 | 800.5 | **3626** | **0** | 131 rounds |

Decision: **KEEP rush-3 (champion).** Min-defense-2 is slightly safer (0 destructions,
better hp_min) but scores lower on mean and combat. Margin is small; revisit if
validation shows champion worst-case matters. Noted as open trade-off.

### Bug found by experiments (fixed)

`brain._mine`: `min(adjacent)` compared `Pos` objects → TypeError when a worker stood
next to 2+ mines of the focused mineral. Production impact: one lost round per
occurrence (server fails closed). Fixed with an explicit sort key; regression via
full test suite (80 tests OK).

### P3-walls — none vs ring-d3 vs ring-d2 (30 games, seeds 1000–1029)

| variant | survival | score | hp_end | destroyed |
|---|---|---|---|---|
| walls-none | 7.1 | 468.8 | 86 | **29/30** |
| walls-ring-d3 (champion) | 9.93 | **804.7** | **3239** | 1 |
| walls-ring-d2 | 9.87 | 794.6 | 2694 | 3 |

Decision: **ring walls are load-bearing; KEEP ring-d3.** Without walls the base dies
in 97% of games. Starting walls day 2 hurts (economy diverted too early).

### P6-sell-timing — full vs 85/70/50% (30 games, seeds 1000–1029)

All variants within noise (804.7 / 804.7 / 804.7 / 805.1). Mine↔vendor trips are too
short in this map for fill thresholds to matter.
Decision: **KEEP sell-when-full (champion); hypothesis REJECTED as irrelevant here.**

### P9-cross-map scoring (sanity, 5 seeds, enemy_target_fraction 0 vs 0.5)

With 50% of waves targeting the enemy team, survival stays 10/10 and the strategy
still harvests enemy-targeted robots that enter weapon range (combat 160 vs the
~133 own-side-only baseline). Robots marching on the enemy base are mostly outside
railgun range — a structural geometry limit, not a policy gap. The -200 threat
deprioritisation keeps own-side threats strictly first (survival gate works).
Decision: **implemented by design; no aggressiveness knob needed. PARTIAL (geometry-capped).**

### P4-upgrade-delay — immediate vs station-only hold (30 games, seeds 1000–1029)

First formulation (delay all vouchers) collapsed: full-HP buildings never triggered
use → 0 upgrades → 17/30 destroyed. Corrected to station-only delay:

| variant | survival | score | upgrades | destroyed |
|---|---|---|---|---|
| immediate (champion) | 9.93 | **804.7** | 5.27 | 1 |
| hold-80 | 9.97 | 797.8 | 3.2 | 1 |
| hold-65 | 9.90 | 790.4 | 3.1 | 2 |
| hold-50 | 9.80 | 778.8 | 3.1 | 3 |

Decision: **REJECT delayed use; KEEP immediate.** The +1500 max HP and extra weapon
levels prevent more damage than the saved heal restores; holding also stalls the
upgrade pipeline (5.3 → 3.2 upgrades/match).

### P7-gold-reserve — 0/25/50 (30 games, seeds 1000–1029)

reserve-0: 804.7 / reserve-25: 790.1 / reserve-50: 784.1; destructions 1/2/3.
Decision: **KEEP reserve-0** (the existing 25g rebuild fund while towers are missing
stays). Hoarding gold delays firepower.

### P11-consumables — DizzyWeapon/Bomb via rocket-cooldown window (30 games)

dizzy-d6 803.2 / dizzy-d4 803.2 / bomb-d6 801.1 vs champion 804.7 — all within noise,
no destroyed-game improvement, hp_end lower (voucher gold diverted).
Decision: **REJECT in synthetic harness.** Scattered synthetic spawns rarely form
3×3 clusters; real clustered waves could differ, but no evidence to justify the
complexity. Code remains available (`StrategyConfig.consumables_enabled`, default off).

### P12-wall-workers — 1 vs 2 stone workers (30 games)

walls-2-workers: 777.2, upgrades 3.1 (economy gutted) vs champion 804.7.
Decision: **KEEP 1 stone worker.**

### P13-upgrade-order — station L3 first vs weapons L3 first (30 games)

weapons-l3-first: 807.6 mean / 263.9 combat vs champion 804.7 / 261.0; same
destructions (1), same p10 (779), hp_end lower (2564 vs 3239, station heal later).
Marginal mean edge → **candidate, decided by validation seeds 2000–2049.**
