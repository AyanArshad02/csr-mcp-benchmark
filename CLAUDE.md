# CLAUDE.md

> Persistent project memory for Claude Code. This file is loaded into context every session.
> Keep it current — when a locked decision changes, change it here too.
> Companion file: `project-state.md` (holds live state + the finish line). This file holds rules +
> locked decisions.

---

## What this project is

**CSR Benchmark** — a research-grade engineering project measuring whether *source attribution*
survives when a RAG knowledge base is consumed by an AI agent (vs. a single-shot RAG call), plus a
thin middleware that recovers lost attribution.

- **Core metric — Citation Survival Rate (CSR), defined by TWO cheap, deterministic checks:**
  For each atomic claim with its cited `source_url`(s):
  1. **Support check (ALCE):** does the cited chunk *entail* the claim? (NLI: premise = chunk,
     hypothesis = claim.)
  2. **Trace-grounding check (faithfulness proxy):** was the cited `source_url` *actually in the
     chunks the agent retrieved across all tool calls* (from the logged trace)?
  CSR = of the gold-supporting `source_url`s the agent **actually retrieved**, what fraction end up
  in the final answer **both** correctly cited (check 2) **and** on a claim the chunk genuinely
  supports (check 1).
- **NO perturbation / counterfactual regeneration in this project.** We take the *concept* from
  Wallat et al. ("correct ≠ faithful"), not its *method*. The trace-grounding check is a deliberate
  necessary-condition proxy for faithfulness. (Full detail: `project-state.md` §2.)
- **Built on:** ALCE citation precision/recall/F1 (NLI entailment) + FActScore-style atomic claim
  decomposition.
- **The boundary under study:** the MCP (Model Context Protocol) tool boundary, where a retrieval
  server hands raw `{source_url, content}` chunks to an agent it doesn't control.
- **Why it exists:** original outreach work aimed at the founders of Kapa AI (Emil Sorensen, CEO;
  Finn Bauer, CTO). Their MCP / Retrieval API / Agent SDK products expose exactly this boundary, and
  nobody measures what happens to citations once a third-party agent re-summarizes the chunks.

**The deliverable is the artifact's correctness, honesty, and reproducibility — not flashy claims.**
Kapa's culture is "accuracy is the only thing that matters." Every number must be reproducible from
`results/`. A clean, honestly-caveated finding beats a dramatic one. An honest *null* result (no
citation drop) is a valid, publishable outcome — never fudge numbers to manufacture a finding.

---

## Hard rules (do not violate)

1. **Never commit secrets.** API keys live in `.env`, which is gitignored. Never print full keys to
   logs. Never hardcode a key in source.
2. **Never fudge, round-to-flatter, or hand-wave a number.** If a result is null/weak/surprising,
   report it honestly. Disqualifying mistake for this project if violated.
3. **The human-verification of the eval set is the user's job, by hand. Do NOT auto-verify gold
   attributions.** You may *draft* candidate Q&A pairs; the user verifies them. The credibility of
   the whole benchmark rests on this.
4. **Pin all dependency versions.** Reproducibility is the point. No floating versions.
5. **Lock decisions, don't drift.** The decisions table below is fixed unless the user explicitly
   changes them. Don't quietly swap the NLI model, citation format, etc.
6. **Verify external APIs against current docs before relying on them.** LangGraph, OpenAI Agents
   SDK, the MCP Python SDK, and Kapa's MCP schema all drift. When a step touches one of these,
   fetch/check the current docs rather than trusting a remembered API shape.
7. **CSR = two cheap checks (support + trace-grounding), NEVER perturbation.** Do not reintroduce
   counterfactual-regeneration scope creep from the Wallat paper. If a plan starts describing
   "altering chunks and re-running," stop — that's out of scope.

---

## Locked decisions

| Decision | Choice | Reason |
|---|---|---|
| Corpus | FastAPI docs (raw Markdown from its GitHub repo) | Clean Markdown, stable public URLs, developer-facing, well-bounded |
| Chunking | Recursive / structure-aware by Markdown headings | Keeps each chunk a coherent citable unit |
| Embedding model | `BAAI/bge-large-en-v1.5` | Strong open model; remember query-prefix on queries only |
| Vector index | Chroma (or FAISS) | Easy persistence |
| MCP tool name | `search_fastapi_knowledge_sources` | Mirrors Kapa's `search_<product>_knowledge_sources` exactly |
| Transport | streamable HTTP | Matches Kapa's hosted server |
| Citation format | `[source_url]` immediately after each claim | Must be IDENTICAL across all 3 pipelines; machine-parseable |
| NLI model (headline) | `cross-encoder/nli-deberta-v3-base` | Fast for thousands of pairs; 3-way output for error analysis |
| NLI model (spot-check) | `google/t5_xxl_true_nli_mixture` on a subset | ALCE-standard model; validates headline numbers |
| Claim decomposition model | A cheap/fast model (free OpenRouter or small OpenAI) | Applied identically across pipelines so biases wash out |
| Entailment decision rule | argmax class = entailment | Simpler and more defensible than a tuned threshold for v1 |
| Eval set size | ~150 human-verified Q&A pairs | Small-but-honest beats large-but-unverified |
| Models under test | 1 frontier model + 1 free OpenRouter model | Shows the effect isn't model-specific without huge cost |
| LLM providers | OpenAI + Anthropic + OpenRouter, behind a strategy pattern | User has keys for all three |
| Agent frameworks | LangGraph + OpenAI Agents SDK | LangGraph has a Kapa-published example; OpenAI SDK does not |
| CSR faithfulness check | Trace-grounding proxy (NOT perturbation) | Cheap, deterministic, defensible necessary-condition; full causal test out of scope |

> A few model-specific choices are still open — see `project-state.md` §6. Fill them into
> `DECISIONS.md` before Phase 1.

---

## Repo structure

```
csr-mcp-benchmark/
├── CLAUDE.md               # this file (rules + locked decisions)
├── project-state.md        # live state + finish line + corrected CSR def (READ §2)
├── DECISIONS.md            # locked-decisions table (mirror of above; update as you go)
├── README.md               # written last (Phase 4/5); leads with thesis + headline number
├── pyproject.toml          # or requirements.txt — PINNED versions
├── .env.example            # template of required keys (NO real keys)
├── .gitignore
├── NOTES.md                # running log of surprises/bugs (raw material for the paper)
├── ingest/                 # OFFLINE: crawl + chunk + embed + index the corpus
├── server/                 # the self-hosted MCP retrieval server (mirrors Kapa's contract)
├── eval_set/               # labeled Q&A dataset (eval.jsonl) + builder script + README
├── pipelines/
│   ├── strategy/           # provider-agnostic LLM strategy-pattern code
│   ├── baseline.py         # single-shot retrieve-then-generate (control condition)
│   ├── langgraph_agent.py  # ReAct agent via langchain-mcp-adapters (Kapa's pattern)
│   └── openai_agent.py     # ReAct agent via OpenAI Agents SDK (MCPServerStreamableHttp)
├── measure/                # decomposition + NLI + trace-grounding + CSR/citation-F1 scoring
│   ├── score.py
│   └── test_scorer.py      # 5 hand-built cases with known expected scores
├── guardrail/              # attribution-guardrail middleware
├── results/                # output tables, per-question traces, logs (every number lives here)
├── b1-notes/               # the 12 study notes + master plan (context library)
└── paper/                  # the writeup (markdown source + final PDF)
```

---

## How to work on this (workflow for Claude Code)

- **One sub-step at a time.** Follow the phases in `b1-notes/B1-MASTER-PLAN.md`; track state in
  `project-state.md`. Do not build multiple phases in one go. Each phase has a checkpoint — treat it
  as a "don't proceed until green" gate.
- **Read the relevant notes before building.** Each master-plan step names the notes that apply
  (in `b1-notes/`). Example: before the server, read `07-building-mcp-server.md` and
  `08-mcp-provenance-contract.md`.
- **The user is the reviewer.** Explain what you built and why, surface the parts that need human
  judgment, and keep code readable enough that the user can defend every line to a sharp CTO.
- **Validate before trusting.** The scorer must pass its 5 hand-built unit cases (Phase 1.3) before
  any real scoring. The server must pass the MCP Inspector test before any agent wiring.
- **Cache aggressively.** Cache LLM + NLI outputs from the start — scoring is re-run many times.
- **Log raw outputs + full traces, not just scores.** The trace-grounding CSR check *requires* the
  full tool-call log, so capturing it is mandatory, not optional.
- **Commit often with honest messages.** Git history is part of the artifact.
- **Update `project-state.md` §1 ("WHERE I AM RIGHT NOW") at the end of each session.**

---

## Cadence

The user is working **daily, ~3–4 hrs/session** (not just weekends). So:
- Prefer **small, completable units per session** — end each session at a green checkpoint or a
  clean commit, never mid-refactor.
- At the start of each session, read `project-state.md` §1 to see where things stand; at the end,
  update it.
- The master plan's "weekend" estimates are just effort sizes — map them to daily sessions (a
  "1 weekend" task ≈ 3–4 daily sessions).

---

## Current status

> Mirror of `project-state.md` §1 — keep both in sync, but `project-state.md` is the source of truth.

- **Phase:** 0 (setup & foundations) — starting now, daily cadence.
- **Next action:** create repo skeleton, fill in `DECISIONS.md`, set up pinned environment, answer
  `project-state.md` §6 open decisions.
- **Blockers:** none. (No real Kapa access yet — self-hosted mirror; "swap in real Kapa" is a later
  env-var change.)

---

## Key references (verify live before relying on API specifics)

- Live state + finish line: `project-state.md`
- Master plan: `b1-notes/B1-MASTER-PLAN.md`
- Study notes index: `b1-notes/00-index.md`
- Kapa hosted MCP contract: https://docs.kapa.ai/integrations/mcp/overview
- Kapa LangGraph example (template for `langgraph_agent.py`): https://github.com/kapa-ai/langchain-agent-example
- Kapa FastMCP proxy example: https://github.com/kapa-ai/fastmcp-proxy-example
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- langchain-mcp-adapters: https://github.com/langchain-ai/langchain-mcp-adapters
- OpenAI Agents SDK MCP docs: https://openai.github.io/openai-agents-python/mcp/
- ALCE (citation metrics): https://github.com/princeton-nlp/ALCE
- FActScore (atomic claims): https://github.com/shmsw25/factscore
- NLI model: https://huggingface.co/cross-encoder/nli-deberta-v3-base
- Embedding model: https://huggingface.co/BAAI/bge-large-en-v1.5
