"""One-call smoke test of each subscription CLI: checks login, sandboxing and JSON parsing.

    python3 scripts/probe_cli.py [claude-code,codex-cli,gemini-cli]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from env import OfficeSearchEnv  # noqa: E402
from harness.agent import SYSTEM, StateTracker, tool_specs  # noqa: E402
from harness.clients import make_client  # noqa: E402

ids = sys.argv[1].split(",") if len(sys.argv) > 1 else ["claude-code", "codex-cli", "gemini-cli"]
cfg = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
env = OfficeSearchEnv()
obs = env.reset(0, "easy")
msgs = [{"role": "system", "content": SYSTEM}, StateTracker(env, obs).message()]
for mid in ids:
    try:
        c = make_client(cfg[mid]).complete(msgs, tool_specs())
        ok = c.tool is not None
        print(f"{'OK ' if ok else 'BAD'} {mid:<12} tool={c.tool} args={c.args} tokens={c.input_tokens}/{c.output_tokens} "
              f"{c.latency_s:.1f}s" + ("" if ok else f"\n    raw: {c.text[:300]!r}"))
    except Exception as e:   # noqa: BLE001 — report every failure mode
        print(f"ERR {mid:<12} {type(e).__name__}: {str(e)[:400]}")
