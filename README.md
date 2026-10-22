# deep-research-agent

Ask a research question, get a cited report. A lead agent splits the work
across child agents that research in parallel, then writes the answer.

## How it works

**Lead** reads the question, picks a mode, hands out the work, and writes the
report. It never searches anything itself.

**Children** do the research. One kind of agent, many running at once, each
with its own context. The lead gives each one a side: neutral, for, or against.
Every child picks its own tools:

| Tool | Source |
|---|---|
| `web` | general web |
| `news` | news articles |
| `papers` | research papers |
| `database` | SQL database, via [NLQuery-agent](https://github.com/naufalkr/NLQuery-agent) over MCP |

**Two modes**, chosen by the lead:

```
Facts   "compare fintech defaults in 3 countries"
        lead -> 1-5 neutral children in parallel, one per part
             -> coverage check: anything unanswered? send more children
             -> report

Debate  "should banks use AI to approve loans?"
        lead -> one child for, one child against, in parallel
             -> rebuttal: each side reads the other's findings and answers them
             -> verdict
```

Before the report is written, every finding is checked against its own
evidence, and a separate pass attaches the citations so they can't be made up.

**Why not one agent?** One agent can loop and search again - it's the baseline
below, with the same four tools. It reads everything in one context and tends
to follow whichever thread it found first. Splitting the work keeps each part
of a question covered, and gives a judgement question a real case on both
sides. It costs more and takes longer, so simple lookups still get a single
child.

## Results

Same questions, same tools, same models (DeepSeek V4), graded by an LLM judge
from 0 to 3 (exact match for database figures). 17 questions; the last 5 of
the 22-question set did not run because the API balance ran out.

| | Questions | Single agent | Multi-agent |
|---|---|---|---|
| Simple lookups | 10 | 2.90 | 2.70 |
| Multi-part | 5 | 3.00 | 2.80 |
| Judgement (debate) | 2 | 2.50 | **3.00** |
| **All** | 17 | **2.88** | 2.76 |
| Passed (score 2+) | | 17/17 | 17/17 |
| Cost / question | | $0.02 | $0.05 |
| Time / question | | 2 min | 8 min |

The lead picked the right mode on all 16 questions that have one.

Honest reading: with the same tools, the single agent is as good on lookups and
multi-part questions, and much cheaper and faster. The debate mode made the
difference on one of the two judgement questions: asked about restricting
paylater, the single agent argued one side and said it had no evidence for the
other. Of the multi-agent's three lost points, one was a bug (a report cut off
mid-sentence, since fixed) and two were answer quality. Seventeen questions,
one run each, so a 0.1 difference is within noise.

Details and earlier versions: [docs/eval-log.md](docs/eval-log.md)

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add DEEPSEEK_API_KEY and EXA_API_KEY

python -m deepresearch ask "your question"            # multi-agent
python -m deepresearch ask "your question" --baseline # single agent
chainlit run ui/app.py                                # web UI
```

Or with Docker:

```bash
docker compose up        # http://localhost:8100
```

The database tool needs [NLQuery-agent](https://github.com/naufalkr/NLQuery-agent)
cloned next to this repo (`../NLQuery-agent`). Without it, everything else
still works.

Eval:

```bash
python -m evals.run_eval --agent baseline --tag baseline
python -m evals.run_eval --agent multiagent --tag multiagent
python -m evals.compare baseline multiagent
```

## Layout

```text
deepresearch/
  graph.py          the run: plan -> children -> second round -> check -> write
  agents/lead.py    plans, checks coverage, writes
  agents/child.py   the research agent and its tools
  agents/verify.py  claim check and citation pass
  agents/baseline.py  single-agent control
  tools/            web search (Exa) and the database bridge (MCP)
  prompts/          one prompt per step
ui/app.py           Chainlit UI
evals/              question set, LLM judge, compare
```
