"""Bundle results + building data for the UI into ui/data.js (works from file:// and http).

    python3 scripts/build_ui_data.py
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.pricing import list_price  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
THUMBS = ROOT / "data" / "thumbs"
OUTCOMES = ["success", "decoy", "bystander_bag", "not_a_bag", "wrong_location", "right_room_wrong_view",
            "gave_up", "budget_exhausted", "disqualified"]


def ci(xs, stat=lambda v: sum(v) / len(v), iters=1000, seed=0):
    """95% bootstrap confidence interval."""
    if len(xs) < 2:
        return [stat(xs), stat(xs)] if xs else [0, 0]
    rng = random.Random(seed)
    samples = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(iters))
    return [samples[int(0.025 * iters)], samples[int(0.975 * iters) - 1]]


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        for diff in (r["difficulty"], "all"):
            groups[(r["agent"]["id"], r["perception"], diff)].append(r)
    rows = []
    for (aid, perc, diff), rs in groups.items():
        info = [r["info"] for r in rs]
        succ = [1.0 if i["success"] else 0.0 for i in info]
        rew = [i["reward"] for i in info]
        cost = [r["usage"].get("cost_usd", 0) for r in rs]
        n_succ = sum(succ)
        a = rs[0]["agent"]
        rows.append({
            "agent": aid, "label": a["label"], "kind": a["kind"], "color": a.get("color"),
            "harness": a.get("harness"), "billing": a.get("billing", "api" if a["kind"] == "model" else None),
            "api_equivalent_usd": sum(r["usage"].get("api_equivalent_usd") or 0 for r in rs) / len(rs),
            "perception": perc, "difficulty": diff, "n": len(rs),
            "success": sum(succ) / len(rs), "success_ci": ci(succ),
            "reward": sum(rew) / len(rs), "reward_ci": ci(rew),
            "process": sum(i["process"] for i in info) / len(rs),
            "efficiency": sum(i["efficiency"] for i in info) / len(rs),
            "budget_used": sum(i["spent"] / i["budget"] for i in info) / len(rs),
            "turns": sum(r["usage"].get("turns", 0) for r in rs) / len(rs),
            "tokens": sum(r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0) for r in rs) / len(rs),
            "cost_per_run": sum(cost) / len(rs),
            "cost_per_success": (sum(cost) / n_succ) if n_succ else None,
            "outcomes": {o: c for o, c in Counter(i["outcome"] for i in info).items()},
        })
    return rows


def slim(r):
    """Run record for replay: drop bulky clue text, keep everything the replay shows."""
    return {"run_id": r["run_id"], "agent": r["agent"], "perception": r["perception"], "difficulty": r["difficulty"],
            "task_id": r["task_id"], "task": r["task"], "info": {k: v for k, v in r["info"].items()},
            "usage": r["usage"], "trajectory": r["trajectory"],
            "answer": {"target": r["answer"]["target"],
                       "decoys": [d for d in r["answer"]["decoys"] if d.get("lookalike", True)],
                       "notes": r["answer"]["notes"], "knowable": r["answer"]["knowable"],
                       "bag_objects": r["answer"].get("bag_objects", {}),
                       "clues": r["answer"].get("clues", {})}}


def thumbs(building):
    THUMBS.mkdir(parents=True, exist_ok=True)
    for fl in building["floors"]:
        for n in fl["nodes"]:
            for v in n["views"]:
                src, dst = ROOT / "data" / v["file"], THUMBS / Path(v["file"]).name
                if not dst.exists():
                    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
                    im.thumbnail((480, 480))
                    im.save(dst, quality=80)
                v["thumb"] = f"data/thumbs/{dst.name}"
                v["src"] = f"data/{v['file']}"
                variants = {role: f"data/images/variants/{v['id']}__{role}.jpg" for role in ("target", "decoy")
                            if (ROOT / "data" / "images" / "variants" / f"{v['id']}__{role}.jpg").exists()}
                if variants:
                    v["variants"] = variants


def main():
    building = json.loads((ROOT / "data" / "building.json").read_text())
    thumbs(building)
    runs = []
    legacy = "--legacy" in sys.argv          # include the original office-floor suite
    for f in sorted(RUNS.glob("*.jsonl")):
        if f.name.endswith("_pilot.jsonl"):
            continue   # smoke-test episodes repeat the first eval task
        if "_dev" in f.name and not f.name.startswith("dryrun"):
            continue   # dev-split scripted runs are for iteration, not the leaderboard
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if f.name.startswith("dryrun_"):   # mock runs must never pool with a real model's runs
                    r["agent"]["id"] = "dryrun_" + r["agent"]["id"]
                if r["agent"]["id"] == "oracle_plan":
                    continue   # sees the answer; a sanity check that tasks are solvable, not a competitor
                if r["agent"]["id"] == "appearance_only":
                    continue   # scripted ablation, kept for the red-team tests but not shown on the page
                if legacy or str(r["difficulty"]).startswith("campus"):
                    runs.append(r)
    # Price subscription rows recorded before list pricing, in memory (never rewrite files a run is appending to).
    models = {m["id"]: m for m in json.loads((ROOT / "config" / "models.json").read_text())["models"]}
    for r in runs:
        m = models.get(r["agent"]["id"])
        if m:
            r["agent"]["label"] = m["label"]      # current display name, even for runs recorded under an older one
        if m and m.get("billing") == "subscription" and not r["usage"].get("cost_usd"):
            r["usage"]["cost_usd"] = list_price(r, m) or 0.0
    bags_path = ROOT / "data" / "bag_labels.json"
    bags = json.loads(bags_path.read_text()) if bags_path.exists() else {"placements": []}
    data = {"building": building, "bags": bags, "leaderboard": aggregate(runs), "runs": [slim(r) for r in runs],
            "outcomes": OUTCOMES, "generated": __import__("time").strftime("%Y-%m-%d %H:%M")}
    out = ROOT / "ui" / "data.js"
    # The page polls ui/stamp.json and refreshes its results when the run set changes.
    stamp = {"runs": len(runs), "ids": hashlib.sha1("".join(r["run_id"] for r in runs).encode()).hexdigest()[:12],
             "generated": data["generated"]}
    stamp_path = ROOT / "ui" / "stamp.json"
    old = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    if "--if-changed" in sys.argv and out.exists() and {k: old.get(k) for k in ("runs", "ids")} == \
            {k: stamp[k] for k in ("runs", "ids")}:
        return
    out.write_text("window.EVAL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    stamp_path.write_text(json.dumps(stamp))
    print(f"{len(runs)} runs, {len(data['leaderboard'])} leaderboard rows → {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
