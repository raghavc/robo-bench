"""Run models through the environment with a hard dollar cap.

    # offline end-to-end test + cost estimate (no API key needed)
    python3 scripts/run_eval.py --dry-run --models all --n 10 --difficulties hard

    # real run (needs .env and config/models.json filled in)
    python3 scripts/run_eval.py --models gpt-6.1-sol,opus-5.5 --n 30 --max-usd 10

Results append to results/runs/model_<id>_<perception>_<split>.jsonl (re-running resumes, skipping done
episodes). Every dollar is logged in results/spend_ledger.json and counts toward --max-usd across runs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from env import Building, OfficeSearchEnv  # noqa: E402
from env.records import RESULTS_DIR, episode_record  # noqa: E402
from env.tasks import campus_task_list, task_list  # noqa: E402
from harness.agent import run_episode  # noqa: E402
from harness.cli_clients import UsageLimitError  # noqa: E402
from harness.clients import MockClient, make_client  # noqa: E402

LEDGER = RESULTS_DIR / "spend_ledger.json"


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--run-tag", default="", help="suffix for a separate result file using the same task split")
    ap.add_argument("--suite", default="campus", choices=["campus", "office"])
    ap.add_argument("--difficulties", default=None, help="default: campus-extreme (campus) / easy,medium,hard (office)")
    ap.add_argument("--n", type=int, default=10, help="tasks per difficulty")
    ap.add_argument("--samples", type=int, default=1, help="runs per task")
    ap.add_argument("--perception", default="oracle", choices=["oracle", "vlm", "rfdetr"])
    ap.add_argument("--max-usd", type=float, default=10.0, help="hard cap on total spend (all runs, from the ledger)")
    ap.add_argument("--dry-run", action="store_true", help="use the mock model; estimate cost with each model's prices")
    args = ap.parse_args()
    load_env()

    cfg = json.loads((ROOT / "config" / "models.json").read_text())
    models = cfg["models"] if args.models == "all" else [m for m in cfg["models"] if m["id"] in args.models.split(",")]
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"total_usd": 0.0, "by_model": {}}
    if not args.dry_run:
        bad = [m["id"] for m in models if m.get("billing") == "api"
               and "TODO" in (m["model"], m["input_per_m"], m["output_per_m"])]
        if bad:
            sys.exit(f"Fill config/models.json for: {', '.join(bad)}")

    b = Building()
    env = OfficeSearchEnv(b, args.perception)
    if args.suite == "campus":
        tasks = campus_task_list(args.split, (args.difficulties or "campus-extreme").split(","), args.n)
    else:
        tasks = task_list(args.split, (args.difficulties or "easy,medium,hard").split(","), args.n)
    estimate = {}
    for m in models:
        if m.get("billing") == "subscription" and args.perception != "oracle" \
                and m.get("provider") not in ("claude_cli", "codex_cli", "agy_cli"):
            print(f"skipping {m['id']}: this CLI adapter is text-only (use --perception oracle)")
            continue
        if m.get("text_only") and args.perception != "oracle":
            print(f"skipping {m['id']}: text-only model (use --perception oracle)")
            continue
        priced = not isinstance(m.get("input_per_m", "TODO"), str)
        prices = {"input_per_m": m["input_per_m"] if priced else cfg["estimate_prices"]["input_per_m"],
                  "output_per_m": m["output_per_m"] if priced else cfg["estimate_prices"]["output_per_m"]}
        client = MockClient() if args.dry_run else make_client(m)
        tag = "dryrun_" if args.dry_run else ""
        suffix = f"_{args.run_tag}" if args.run_tag else ""
        out = RESULTS_DIR / "runs" / f"{tag}model_{m['id']}_{args.perception}_{args.split}{suffix}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["run_id"] for l in out.read_text().splitlines()} if out.exists() else set()
        spent_here = 0.0
        limit_hit = False
        for diff, seed in tasks:
            if limit_hit:
                break
            for sample in range(args.samples):
                agent = {"id": m["id"], "label": m["label"] + (" (mock)" if args.dry_run else ""),
                         "kind": "mock" if args.dry_run else "model", "model": m.get("model"),
                         "color": m.get("color"), "harness": m.get("harness", "lost-found harness"),
                         "billing": m.get("billing", "api")}
                obs = env.reset(seed, diff, nonce=f"{m['id']}:{sample}")
                if args.perception != "oracle" and not env.vision_ready():
                    continue   # skip tasks whose bag photos haven't been captured yet
                run_id = f"{m['id']}__{args.perception}__{diff}__{env.s.id}__{sample}"
                if run_id in done:
                    continue

                billed = m.get("billing", "api") == "api"      # subscriptions report list price but cost nothing

                def guard(ep_cost):
                    return args.dry_run or not billed or ledger["total_usd"] + ep_cost < args.max_usd

                if billed and not guard(0):
                    print(f"Spend cap ${args.max_usd} reached (ledger ${ledger['total_usd']:.2f}); stopping.")
                    return finish(estimate, ledger, args)
                info = None
                try:
                    for attempt in (1, 2):      # a CLI timeout or transient error retries the episode once
                        try:
                            info, usage = run_episode(env, obs, client, prices, guard)
                            break
                        except UsageLimitError:
                            raise
                        except RuntimeError as e:
                            print(f"{m['id']}: {env.s.id} attempt {attempt} failed: {str(e)[:160]}")
                            obs = env.reset(seed, diff, nonce=f"{m['id']}:{sample}")
                except UsageLimitError as e:
                    print(f"\n{m['id']}: usage limit hit — {e}\nRe-run the same command after it resets; "
                          f"finished episodes are kept and skipped.")
                    limit_hit = True
                    break
                if info is None:
                    continue           # not recorded, so a later re-run tries this task again
                usage["sample"] = sample
                rec = episode_record(env, info, agent, usage)
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                spent_here += usage["cost_usd"]
                if not args.dry_run and billed:
                    ledger["total_usd"] = round(ledger["total_usd"] + usage["cost_usd"], 6)
                    ledger["by_model"][m["id"]] = round(ledger["by_model"].get(m["id"], 0) + usage["cost_usd"], 6)
                    LEDGER.write_text(json.dumps(ledger, indent=1))
                print(f"{m['id']:<12} {diff:<6} {env.s.id} {info.get('outcome', '?'):<16} "
                      f"reward={info.get('reward', 0):+.2f} turns={usage['turns']:<3} ${usage['cost_usd']:.4f}")
        estimate[m["id"]] = spent_here
    finish(estimate, ledger, args)


def finish(estimate, ledger, args):
    label = "Estimated (mock tokens × prices)" if args.dry_run else "Spent this run"
    print(f"\n{label}: " + ", ".join(f"{k} ${v:.3f}" for k, v in estimate.items()) +
          f"  | total ${sum(estimate.values()):.3f}")
    if not args.dry_run:
        print(f"Ledger total: ${ledger['total_usd']:.2f} of ${args.max_usd:.2f} cap")


if __name__ == "__main__":
    main()
