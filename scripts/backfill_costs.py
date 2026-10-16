"""Write list-price costs into finished subscription run files (build_ui_data.py also prices them in memory).

Pass file names to limit it; skip files a running eval is still appending to. Only rows with cost_usd 0 change."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.pricing import CODEX_CACHED_SHARE, list_price  # noqa: E402

models = {m["id"]: m for m in json.loads((ROOT / "config/models.json").read_text())["models"]}
only = set(sys.argv[1:])
for f in sorted((ROOT / "results/runs").glob("model_*.jsonl")):
    if only and f.name not in only:
        continue
    rows = [json.loads(l) for l in f.open()]
    changed = 0
    for r in rows:
        m = models.get(r["agent"]["id"])
        u = r["usage"]
        if not m or m.get("billing") != "subscription" or u.get("cost_usd"):
            continue
        cost = list_price(r, m)
        if cost is None:
            continue
        if m["provider"] == "codex_cli" and not u.get("api_equivalent_usd"):
            u["input_tokens"] = round(u["input_tokens"] / (1 + CODEX_CACHED_SHARE))
            u["cost_estimated"] = True
        u["cost_usd"] = u["api_equivalent_usd"] = round(cost, 6)
        changed += 1
    if changed:
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{f.name}: priced {changed} rows")
