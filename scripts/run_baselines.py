"""Run the scripted reference agents and print a results table; writes results/runs/scripted_*.jsonl.

    python3 scripts/run_baselines.py [--split dev|eval] [--n 200]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import Building, OfficeSearchEnv  # noqa: E402
from env.baselines import CAMPUS_AGENTS, REFERENCE_AGENTS  # noqa: E402
from env.records import append, episode_record  # noqa: E402
from env.tasks import campus_task_list, task_list  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", default="eval")
p.add_argument("--n", type=int, default=None)
p.add_argument("--suite", default="campus", choices=["campus", "office"])
args = p.parse_args()

b = Building()
assert b.is_connected(), "building graph is disconnected"
env = OfficeSearchEnv(b)
LABELS = {"random": "Random walk", "exhaustive": "Brute-force sweep", "clue_reasoner": "Clue reasoner",
          "oracle_plan": "Oracle plan (privileged)", "appearance_only": "Appearance only"}
AGENTS = CAMPUS_AGENTS if args.suite == "campus" else REFERENCE_AGENTS
TASKS = (campus_task_list(args.split, ("campus-extreme", "campus-hard"), args.n) if args.suite == "campus"
         else task_list(args.split, limit=args.n))
print(f"{'difficulty':<8} {'agent':<14} {'success':>8} {'reward':>7} {'proc':>5} {'eff':>5} {'spent':>6}  failure modes")
for name, cls in AGENTS.items():
    recs = []
    for diff, seed in TASKS:
        obs = env.reset(seed, diff, nonce=name)
        agent = cls(seed) if name == "random" else cls()
        info = agent.run(env, obs)
        recs.append(episode_record(env, info, {"id": name, "label": LABELS[name], "kind": "scripted"}))
    for diff in sorted({r["difficulty"] for r in recs}):
        rs = [r["info"] for r in recs if r["difficulty"] == diff]
        k = len(rs)
        oc = Counter(i["outcome"] for i in rs)
        fails = ", ".join(f"{o}:{c}" for o, c in oc.most_common() if o != "success")
        print(f"{diff:<8} {name:<14} {sum(i['success'] for i in rs)/k:>8.0%} {sum(i['reward'] for i in rs)/k:>7.3f} "
              f"{sum(i['process'] for i in rs)/k:>5.2f} {sum(i['efficiency'] for i in rs)/k:>5.2f} "
              f"{sum(i['spent'] for i in rs)/k:>6.1f}  {fails}")
    append(recs, f"scripted_{args.suite}_{name}_{args.split}")
