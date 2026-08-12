# Eval log

What each measured run showed, and what it changed. Kept because several runs
looked like progress and were not.

Results live in `runs/evals/<tag>.json`.

---

## v0.1-baseline — single agent, 2 tools

| Set | Passed | Mean | $/q | s/q |
|---|---|---|---|---|
| Simple (10) | 9/10 | 2.70 | $0.0028 | 43 |
| Hard (5) | 5/5 | 2.60 | $0.0196 | 122 |

The one simple-question failure (`contradiction`) asserted the opposite of what
its sources said — nothing in a single-agent loop cross-checks a claim.

Both hard-question gaps were **coverage, not accuracy**: `hard-lending-compare`
covered Indonesia in depth and left Thailand without a figure;
`hard-mcp-timeline` missed a stage. A sequential agent spends its attention on
whichever thread it followed first.

---

## v0.2-multiagent — lead + parallel subagents

| Set | Passed | Mean | $/q | s/q |
|---|---|---|---|---|
| Simple (10) | 10/10 | 2.90 | $0.0101 | 117 |
| Hard (5) | 5/5 | 3.00 | $0.0295 | 304 |

Overall mean 2.67 → 2.93, at ~2x cost and ~2.5x wall time. No regression on
simple questions, which was the risk worth checking — spending five subagents
on a lookup could easily have scored worse.

### The first multi-agent build scored 1.60 — below the baseline

Parallel subagents alone made things worse. The lead planned once and never
checked whether the plan covered the question, so a part it overlooked stayed
missing however many subagents ran. The failing answers said so themselves:

> *"they lack any specific data on Indonesia's default figures... A direct
> three-way comparison is therefore not possible."*

The baseline could adapt mid-run — search, see what came back, search again.
The multi-agent build could not. Adding a **coverage review** between gathering
and writing (lead compares findings to the question, spends another round on
whatever nothing answered) is what took it from 1.60 to 3.00.

---

## Runs that looked like progress and were not

| Reported | Actually |
|---|---|
| mean 2.40, "5/5 passed" | Three answers had **zero findings**. Subagents timed out at 120s; the judge rubric gave 2/3 for admitting it honestly. |
| One case 0/3 | `synthesize` returned the review step's JSON instead of prose — all three lead steps shared one prompt file with three output formats. |
| Header showed no agent | A `str.replace()` patch silently matched nothing. |

Two of these read as success. Believing the first would have meant reporting
"coverage review works, 1.60 → 2.40" while three of five answers were empty.

Each was caught by instrumentation rather than by reading code:

- `result.failures` — named the timeouts immediately
- `TruncatedResponse` — turns a silently empty reply into a stated cause
- storing the plan in eval results — showed the planner had made one subagent
  for a question needing four

The judge rubric was also wrong, not just the code: "honest uncertainty scores
2, not 0" was written for an answer that is mostly right and admits one hole.
It now reads *"an answer that gathered nothing and says so is a 0 however
gracefully it says it."*

---

## Running a sweep

Sweeps take 45-70 minutes with verification on, so they are usually
backgrounded. Two things make the output readable while it runs:

```bash
python -m evals.run_eval --agent verified --tag v0.3 2>&1 \
  | grep --line-buffered -vE "INFO|HTTP Request|server\.py|_client\.py"
```

- **Keep stderr.** Redirecting it to `/dev/null` to hide the MCP INFO lines also
  discards tracebacks: one sweep died on an unhandled `TruncatedResponse` and
  the output file held no clue why.
- **`--line-buffered`.** Without it grep holds every line in its own buffer and
  the file stays empty until the sweep ends, which looks identical to a hang.

## Known issues

- ~~**`plan_queries` truncates on `deepseek-v4-flash`.**~~ Fixed: `PLAN_TOKENS`
  raised to 6000.
- **A 15-question sweep run across a machine sleep is not comparable.** One case
  took 2397s against 304s in a clean run, and the stalls produced timeouts that
  are artefacts, not behaviour. The hard-question figures above come from the
  clean run; discard sweeps whose per-question times jump like that.
