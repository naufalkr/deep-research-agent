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

The flow is a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`:
each step is a node, the children run in parallel through `Send`, and the
mode decides whether another round runs. Agents, prompts and tools are plain
Python - no LangChain.

## Results

22 questions, same tools and models (DeepSeek V4) for both, graded by an LLM
judge from 0 to 3 (exact match for database figures). The single agent is the
baseline: one agent, one loop, the same four tools.

| | Questions | Single agent | Multi-agent |
|---|---|---|---|
| Simple lookups | 10 | 2.90 | 2.70 |
| Multi-part | 5 | 3.00 | 2.80 |
| News + papers | 2 | 1.50* | 2.50 |
| Judgement (debate) | 5 | 2.80 | 2.80 |
| **All** | 22 | 2.77 | 2.73 |
| Passed (score 2+) | | 21/22 | 22/22 |
| Cost / question | | $0.02 | $0.06 |
| Time / question | | 2 min | 8.5 min |

The lead picked the right mode on all 21 questions that have one.

Honest reading: with the same tools, one agent is about as good and much
cheaper and faster - multi-agent did not beat it here. One run per question,
so differences of 0.1-0.2 are noise. Two debate answers came back empty from
the API and were rerun after a fix (empty replies are now retried).

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
  graph.py          LangGraph flow: plan -> children -> next round -> check -> write
  agents/lead.py    plans, checks coverage, writes
  agents/child.py   the research agent and its tools
  agents/verify.py  claim check and citation pass
  agents/baseline.py  single-agent control
  tools/            web search (Exa) and the database bridge (MCP)
  prompts/          one prompt per step
ui/app.py           Chainlit UI
evals/              question set, LLM judge, compare
```
