# deep-research-agent

Ask one open-ended research question and get a cited report back. A lead agent
plans the research, spawns **subagents that run in parallel**, a critic verifies
every claim, and a separate pass attaches the citations.

Most deep research agents only read the web. This one also **queries a real
database** — when a claim needs a hard number, a subagent runs SQL against it
instead of trusting an article.

**Loop:** `plan → delegate (parallel) → research (web + SQL) → verify → synthesize → cite`

---

## Status

Work in progress. Built in phases, tagged as each one lands:

| Tag | Phase | State |
|---|---|---|
| `v0.1-baseline` | Single agent, 2 tools — the control group | ☐ |
| `v0.2-multiagent` | Lead agent + parallel subagents | ☐ |
| `v0.3-verified` | Critic + separate citation pass | ☐ |
| `v0.4-evaluated` | Eval harness: accuracy, cost, latency | ☐ |
| `v0.5-ui` | Web UI with streaming + trace view | ☐ |
| `v1.0` | Memory, export, Docker | ☐ |

Phase 1 exists to prove the rest earns its cost — every later tag is measured
against it.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your API keys

python -m deepresearch ask "your research question here"
```

---

## Notes

- **Orchestrator-worker** — a lead agent plans and delegates; subagents each get
  their own context window, so deep exploration never pollutes the main thread.
- **Web + database** — `web_search` for context and recent events,
  `query_database` (via [nlquery-agent](https://github.com/naufalkr/NLQuery-agent)
  over MCP) for numbers that have to be right.
- **Citations are a separate pass** — synthesis writes the narrative, a second
  agent attaches sources. Asking one agent to do both gets you invented
  citations.
- **Model tiering per role** — a frontier model plans and verifies; cheaper
  open-weight models do extraction and summarization. Configured per agent in
  `.env`.
- **Effort scales to the question** — a simple lookup gets one subagent, a
  complex analysis gets five. Multi-agent costs ~15x the tokens of a chat turn,
  so this is the main cost control.
- **Evaluated, not vibed** — a fixed question set plus an LLM judge scores every
  change to accuracy, cost, and latency.

Design rationale, architecture, and best practices: [docs/project-overview.md](docs/project-overview.md)

## Project structure

```text
.
├── deepresearch/              # core engine
│   ├── __main__.py            #   python -m deepresearch
│   ├── cli.py                 #   ask / eval
│   ├── config.py              #   per-role model settings from .env
│   ├── llm.py                 #   provider-agnostic client
│   ├── observability.py       #   token / latency / cost tracking
│   ├── state.py               #   shared research state
│   ├── graph.py               #   LangGraph wiring
│   ├── agents/
│   │   ├── lead.py            #     plan → delegate → synthesize  ★
│   │   ├── researcher.py      #     web subagent
│   │   ├── analyst.py         #     database subagent
│   │   ├── critic.py          #     claim verification
│   │   └── citer.py           #     citation pass
│   ├── tools/
│   │   ├── web.py             #     search + fetch
│   │   └── database.py        #     bridge to nlquery-agent
│   └── prompts/               #   agent prompts as files
│
├── api/main.py                # FastAPI + SSE streaming
├── run_api.py                 # web API entry point
├── mcp_server.py              # MCP server
│
├── web/                       # frontend (Vite + React + Tailwind)
├── evals/                     # question set + LLM judge + metrics
├── tests/                     # pytest suite
└── docs/project-overview.md   # design rationale
```
