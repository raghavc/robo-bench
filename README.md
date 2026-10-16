# Lost & Found: an office-search RL environment

An agent must find a coworker's lost backpack in a real office building (photos captured on two floors),
reasoning from a calendar, badge log, team chat and notes taped inside rooms, under a search budget.
Built on Roboflow (labeling, dataset, auto-label models; RF-DETR detector planned for the perception step).

```
data/         photos, building map, Roboflow labels, capture plan        (data/README.md)
env/          scenario generator, environment + verifier, scripted agents
harness/      model loop (compact state, cost tracking) and API clients
scripts/      run_baselines.py · run_eval.py · build_ui_data.py
tests/        test_verifier.py — red-team suite (agents that try to game the reward)
config/       models.json — model slugs + prices
ui/           data.js (generated) + charts.js
index.html    benchmark page: leaderboard, Pareto, checkpoint chart, example runs with steps
```

## Campus suite (the benchmark)

Floors 3–6 are a real Berkeley capture with one black Databricks backpack photographed in 11 places, in four
configurations (plain / cap / keychain / cap + keychain). `env/campus.py` builds each task backwards from one of
those bags and plants clues so exactly one bag survives: split variant facts, a "had it → noticed it gone" window,
**restricted floors** (look-alikes on floors the owner's badge can't open), coworker claims, early sightings,
cancelled meetings, chains of taped notes, and a wrong guess. Every task is re-verified before it is accepted.

```bash
python3 tests/test_campus.py                       # 600 tasks: unique answer, solvable, red-team
python3 scripts/run_baselines.py                   # oracle plan / appearance-only / random on the frozen 30+30
python3 scripts/run_eval.py --models all --n 30    # models on 30 campus-extreme tasks (12-step cap)
```

Grader output per episode: `reward_outcome` (0/1, the RL signal), `reward_shaped`, and a binary `rubric`
(clues_first, badge_access, followed_notes, avoided_traps, verified_before_report, no_inspect_spam).

## Run it

```bash
python3 tests/test_verifier.py                 # 10 anti-gaming checks
python3 scripts/run_baselines.py               # scripted reference agents on the frozen eval set
python3 scripts/run_eval.py --dry-run --n 10   # full pipeline with a mock model + cost estimate (no key)
python3 scripts/build_ui_data.py               # bundle results for the UI
python3 -m http.server 8000                    # then open http://localhost:8000
```

## Running real models (≤ $10)

1. `cp .env.example .env` and add `OPENROUTER_API_KEY` (one key covers GPT, Claude, Gemini, Kimi).
2. In `config/models.json`, set each model's router slug and its per-million-token prices.
3. Pilot first: `python3 scripts/run_eval.py --models all --n 2 --difficulties hard --max-usd 1`
4. Main run: `python3 scripts/run_eval.py --models all --n 10 --max-usd 10`
   `--max-usd` is a hard cap across all runs (tracked in `results/spend_ledger.json`); re-running resumes.
5. `python3 scripts/build_ui_data.py` and refresh the page.

## Running on subscriptions instead of API credits

`config/models.json` has three subscription entries that drive each vendor's own CLI as a model proxy.
They are text-only (`--perception oracle`) and labelled with their harness on the leaderboard.

| Entry | Needs | Status |
|---|---|---|
| `claude-code` | Claude Code logged in (Pro/Max) | tested; all built-in tools disabled, our system prompt replaces the default |
| `codex-cli` | `codex login` with ChatGPT | tested; runs with `--ignore-user-config` and memories off, so no saved context leaks in; adds ~19k tokens/turn of its own prompt |
| `gemini-cli` | `GEMINI_API_KEY` from AI Studio | personal Google sign-in is no longer accepted by Gemini CLI; an API key works with the same adapter |

```bash
python3 scripts/probe_cli.py                                   # one call each: login + parsing check
python3 scripts/run_eval.py --models claude-code,codex-cli --n 10
```

Each call runs in an empty temp directory. Codex and Gemini can't have their tools fully switched off, so
their event logs are checked: any shell/file/tool use disqualifies the episode (outcome `disqualified`,
reward 0). When a plan's usage limit is hit the run stops cleanly; re-run the same command later to resume.
Expect ~30–50 s per episode. Cost is still reported, as tokens × the vendor's public per-token API price
(`input_per_m` / `cached_input_per_m` / `output_per_m` in `config/models.json`), but it does not count against
`--max-usd`. `scripts/backfill_costs.py` prices runs recorded before this. Results compare *model + vendor harness*, not models on identical terms.

## Task and scoring

* **Tools:** `read_calendar`, `read_chat`, `read_badge_log`, `go_to`, `search`, `inspect`, `report`, `give_up`.
  Each costs budget; changing floors costs more. Budgets per difficulty are calibrated so the scripted
  clue reasoner usually fits and a full sweep of the building does not.
* **Difficulty:** hard adds cancelled meetings still on the calendar, room changes posted only on a note
  inside the original room, a misleading guess from the owner, and look-alike decoys placed in the rooms
  the clues point to. Real bystander bags from the photos appear in every task.
* **Reward:** `0.5 + 0.25·efficiency + 0.25·process` on success, `−0.25` for reporting the wrong bag,
  `0` for giving up or running out of budget.
* **Process score:** every step is graded against what the clues justified (never against the answer),
  so a lucky guess or brute force scores below genuine reasoning.
* **Anti-gaming:** one report per episode; only inspected bags can be reported; object ids are random per
  episode; observations never contain the answer or file names; invalid calls cost budget.

## Perception modes

| Mode | What the model gets | Status |
|---|---|---|
| `oracle` | Objects from Roboflow auto-labels, bag details as text | ready |
| `vlm` | The four photos of each searched room | needs backpack photos (`data/capture_plan.md`) |
| `rfdetr` | Objects listed, bag details only from the photo | needs backpack photos; object lists still come from auto-labels until an RF-DETR model is trained and wired in |

## Adding floors

Add the floor to `data/building.json` (rooms, views, neighbors, `tower_floor`), add the photos to
`data/images/`, link it with `vertical_links`, and add any new venues to `VENUES` in `env/scenario.py`.
Then re-run the calibration (`run_baselines.py`), since budgets depend on building size.
