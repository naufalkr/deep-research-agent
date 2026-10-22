# Eval log

What each measured run showed, and what it changed. Kept because several runs
looked like progress and were not. Results live in `runs/evals/<tag>.json`.

---

## v1.0 - debate mode, tools per child

The lead now picks a mode. **Facts**: neutral children, one per part, plus
coverage rounds. **Debate**: a child for and a child against, then a rebuttal
round. Every child picks its own tools (web, news, papers, database). The
single-agent baseline got the same four tools, so the comparison is about the
architecture, not the tools.

17 questions (the last 5 of the 22 did not run - the API balance ran out):

| | n | Single agent | Multi-agent |
|---|---|---|---|
| Simple lookups | 10 | 2.90 (10/10) | 2.70 (10/10) |
| Multi-part | 5 | 3.00 (5/5) | 2.80 (5/5) |
| Judgement (debate) | 2 | 2.50 (2/2) | 3.00 (2/2) |
| **All** | 17 | **2.88** | 2.76 |
| Cost / question | | $0.02 | $0.05 |
| Time / question | | 122s | 482s |

Mode chosen correctly: 16/16.

**What it says.** The earlier multi-part gap (v0.1 left a country thin) did not
show up this time: with news and papers tools the single agent covered every
part. The debate mode helped where it was meant to - on paylater, the single
agent argued one side and said it had no evidence for the other; the debate
gave both sides and a verdict. Multi-agent costs ~2.4x and takes ~4x longer.

Where multi-agent lost points:

| Case | Score | Why |
|---|---|---|
| `web-mcp` | 2 | **Bug.** The report was cut off mid-sentence: the reasoning model used most of its token budget thinking. Only empty replies were retried. Now any cut-off reply is retried with a bigger budget. |
| `hard-model-landscape` | 2 | Claude was left unsupported; the coverage check did not flag it as a gap. |
| `unanswerable` | 2 | Said revenue was "$0.00" instead of "no data for that period". |

Caveats: 17 questions, one run each, judged by the same model family that
answered. Differences of 0.1-0.2 are within noise.

### Found while running it

| Looked like | Actually |
|---|---|
| `db-revenue` failed, 1/3 | The answer was right. It wrote "United States"; the exact-match check only accepted "USA". Checks now accept alternatives. |
| Baseline `sources-lending` 0/3, "fabricated" | The judge assumed today was its training cutoff and called real 2026 news invented. It is now told the date. This question is not among the 17 compared above. |
| A budget stop restarted the run | Errors inside the database session came back wrapped by the MCP task groups and were taken for "database down", so the run started again without it. Fixed. |
| Cost estimate too low | Tokens from truncated-and-retried calls were never counted. Fixed, but it is why this sweep ran out of balance. |
| Mock results in the real search cache | Tests wrote fake results into `runs/cache`. 23 files removed; tests now use a temp dir. |

---

## v0.2-multiagent - lead + parallel subagents

| Set | Passed | Mean | $/q | s/q |
|---|---|---|---|---|
| Simple (10) | 10/10 | 2.90 | $0.0101 | 117 |
| Hard (5) | 5/5 | 3.00 | $0.0295 | 304 |

Against v0.1 (2 tools, no news or papers): overall 2.67 -> 2.93, at ~2x cost
and ~2.5x wall time.

### The first multi-agent build scored 1.60 - below the baseline

Parallel subagents alone made things worse. The lead planned once and never
checked whether the plan covered the question, so a part it overlooked stayed
missing however many subagents ran. The failing answers said so themselves:

> *"they lack any specific data on Indonesia's default figures... A direct
> three-way comparison is therefore not possible."*

Adding a **coverage review** between gathering and writing (the lead compares
findings to the question and spends another round on whatever nothing
answered) took it from 1.60 to 3.00.

---

## v0.1-baseline - single agent, 2 tools

| Set | Passed | Mean | $/q | s/q |
|---|---|---|---|---|
| Simple (10) | 9/10 | 2.70 | $0.0028 | 43 |
| Hard (5) | 5/5 | 2.60 | $0.0196 | 122 |

Both hard-question gaps were coverage, not accuracy: `hard-lending-compare`
left Thailand without a figure, `hard-mcp-timeline` missed a stage.

---

## Runs that looked like progress and were not

| Reported | Actually |
|---|---|
| mean 2.40, "5/5 passed" | Three answers had **zero findings**. Subagents timed out; the judge gave 2/3 for admitting it honestly. |
| One case 0/3 | The report step returned the review step's JSON - three lead steps shared one prompt file. |
| Header showed no agent | A `str.replace()` patch silently matched nothing. |

Each was caught by instrumentation rather than by reading code: failures
recorded per child, `TruncatedResponse` naming an empty reply, and the plan
stored with every eval result. The judge rubric now reads *"an answer that
gathered nothing and says so is a 0 however gracefully it says it."*

## Running a sweep

```bash
python -m evals.run_eval --agent multiagent --tag v1.0 2>&1 \
  | grep --line-buffered -vE "INFO|HTTP Request|server\.py|_client\.py"
```

Keep stderr (it carries the tracebacks), and keep `--line-buffered` or the log
stays empty until the sweep ends. Results are saved after every question. A
sweep that ran across a machine sleep is not comparable - discard it.
