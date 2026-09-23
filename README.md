# Competition — stable delivery

Current strategy and evidence: [FINAL_REPORT.md](FINAL_REPORT.md).
All overnight experiments and logs are preserved; their simulator is not the official judger.

## Run

Submit the complete `Demo/CoreGeek/CoreGeek` directory (including `src` and `layout.json`).
Requires Python >=3.11, standard library only; no runtime installation is needed.

```bash
cd Demo/CoreGeek/CoreGeek
bash run.sh 8000
# Or on Windows / without bash:
python -B main3.py 8000
```

Listens on `0.0.0.0:<port>`. POST JSON returns `roleCommandMap`, `prompt`, `executeCmd`.
Set `PYTHON` to an interpreter path for run.sh when python3 is not the desired interpreter.
Set `AGENT_LAYOUT` to an absolute JSON file path to use independently verified building sites.
Default layout contains sample evidence only; inferred fallback sites still need official validation.

## Verify

```bash
cd Demo/CoreGeek/CoreGeek
python -B -m unittest discover -s tests -v
python -m compileall -q src main3.py
```

Production selects `brain.champion_config()` (two day-one railguns, then the rocket).
`StrategyConfig()` intentionally remains the historical rush baseline for archived experiments.
Task/LLM/treasure and experimental consumable purchasing remain disabled.
