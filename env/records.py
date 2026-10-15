"""Episode records shared by scripted and model runs. One JSON object per line in results/runs/*.jsonl."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .office_env import OfficeSearchEnv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def episode_record(env: OfficeSearchEnv, info: dict, agent: dict, usage: dict | None = None) -> dict:
    s = env.s
    return {
        "run_id": f"{agent['id']}__{env.perception}__{s.difficulty}__{s.id}__{usage.get('sample', 0) if usage else 0}",
        "agent": agent,                       # {"id", "label", "kind": scripted|model|exploit, "model"?, "provider"?}
        "perception": env.perception,
        "difficulty": s.difficulty,
        "task_id": s.id,
        "seed": s.seed,
        "info": info,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "turns": len(env.trajectory) - 1},
        "trajectory": env.trajectory,
        "task": s.public_task(),
        "answer": {"target": s.backpack, "decoys": s.decoys, "notes": s.notes, "knowable": s.knowable,
                   "clues": s.clues,
                   # hidden roles of every bag object id in this episode (for checkpoint analysis)
                   "bag_objects": {oid: o["bag"] for oid, o in env.objects.items() if o["bag"]}},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def append(records: list[dict], name: str) -> Path:
    path = RESULTS_DIR / "runs" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path
