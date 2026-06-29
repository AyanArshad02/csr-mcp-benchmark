
# DECISIONS.md

> Locked decisions for the CSR benchmark.
> **Do not change anything here mid-experiment** — changing any row means re-running every
> pipeline that depended on it. If you must change something, add a dated note explaining why.
> Last updated: 2025-06-28

---

## Core pipeline decisions (locked from CLAUDE.md)

| Decision                 | Choice                                                                                                                  | Reason                                                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Corpus                   | FastAPI docs — raw Markdown from the official GitHub repo (`tiangolo/fastapi`, `docs/en/docs/`)                    | Clean Markdown, stable public URLs, developer-facing, well-bounded (~hundreds of pages)                                                                              |
| Chunking strategy        | Recursive / structure-aware by Markdown headings                                                                        | Keeps each chunk a coherent citable unit; matches how real doc tools chunk                                                                                           |
| Embedding model          | `BAAI/bge-large-en-v1.5`                                                                                              | Strong open model.**CRITICAL:** apply query-prefix `"Represent this sentence for searching relevant passages: "` on queries only — NOT on indexed documents |
| Vector index             | Chroma with disk persistence                                                                                            | Easy persistence; can rebuild from`ingest/` scripts                                                                                                                |
| MCP tool name            | `search_fastapi_knowledge_sources`                                                                                    | Mirrors Kapa's`search_<product>_knowledge_sources` convention exactly                                                                                              |
| MCP transport            | streamable HTTP                                                                                                         | Matches Kapa's hosted server transport                                                                                                                               |
| Citation format          | `[source_url]` immediately after each claim, e.g. `FastAPI uses Pydantic [https://fastapi.tiangolo.com/features/].` | Must be IDENTICAL across all 3 pipelines; machine-parseable with a simple regex                                                                                      |
| NLI model (headline)     | `cross-encoder/nli-deberta-v3-base` (HuggingFace)                                                                     | Fast enough for thousands of pairs on CPU/small GPU; 3-way output (entailment / contradiction / neutral) for error analysis                                          |
| NLI model (spot-check)   | `google/t5_xxl_true_nli_mixture` on a ~30-item subset                                                                 | The ALCE-paper-standard model; used to cross-validate headline numbers, not as primary scorer                                                                        |
| Entailment decision rule | `argmax class == "entailment"` → supported                                                                           | Simpler and more defensible than a tuned probability threshold for v1                                                                                                |
| Eval set size            | ~150 human-verified Q&A pairs                                                                                           | Small-but-honest beats large-but-unverified; human verification is the credibility anchor                                                                            |
| Eval set variety         | Mix of single-source, multi-source (2-3 URLs), and unanswerable questions; mixed difficulty                             | All-easy set falsely "disproves" the thesis; multi-source questions are the most informative                                                                         |
| Agent frameworks         | LangGraph + OpenAI Agents SDK                                                                                           | LangGraph: Kapa publishes a reference example. OpenAI Agents SDK: no Kapa template, follows OpenAI MCP docs directly                                                 |
| CSR faithfulness check   | **Trace-grounding proxy — NOT perturbation**                                                                     | Cheap, deterministic, defensible. Full Wallat-style counterfactual testing (alter chunks, re-run) is out of scope. See`project-state.md` §2                       |

---

## Model decisions (locked 2025-06-28)

| Decision                            | Choice                                                  | Reason                                                                                                                |
| ----------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Frontier model**            | `gpt-4o-mini` (OpenAI)                                | Strong, affordable, well-documented; compatible with LangGraph and OpenAI Agents SDK natively                         |
| **Free / second model**       | `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) | Free tier; large model for quality comparison; shows CSR effect isn't model-specific                                  |
| **Claim decomposition model** | `gpt-4o-mini` (same as frontier)                      | Applied identically across all pipelines so any decomposition bias washes out in relative comparisons; cheap and fast |

---

## Infrastructure decisions (locked 2025-06-28)

| Decision                      | Choice                                                | Reason                                                                                                                                                                       |
| ----------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Repo visibility**     | Public from day one                                   | Cleaner commit history; nothing to hide; founders can see the work in progress                                                                                               |
| **Notion thinking-log** | Yes — kept in parallel with`NOTES.md`              | NOTES.md = raw technical surprises and bugs. Notion = first-principles thinking, predictions before runs, what I'd do differently. Both feed the paper's methodology section |
| **Python env manager**  | `uv`                                                | Faster than pip; lockfile (`uv.lock`) pins exact versions for reproducibility                                                                                              |
| **LLM output caching**  | Disk cache from day one (`.cache/` dir, gitignored) | Scoring is re-run many times; caching saves money + protects against flaky OpenRouter availability                                                                           |

---

## What each model is used for

```
gpt-4o-mini (OpenAI)
  → Frontier model under test (all 3 pipelines: baseline, LangGraph, OpenAI SDK)
  → Claim decomposition (applied identically to all pipeline outputs)

nvidia/nemotron-3-ultra-550b-a55b:free (OpenRouter)
  → Free model under test (baseline + LangGraph pipelines only, via strategy pattern)
  → Shows the citation-drop effect isn't specific to OpenAI models

cross-encoder/nli-deberta-v3-base (local HuggingFace)
  → NLI engine for ALL citation scoring (Check 1: support)
  → Also used inline by the guardrail middleware

google/t5_xxl_true_nli_mixture (HuggingFace, spot-check only)
  → Cross-validation on ~30 items to confirm DeBERTa numbers aren't anomalous
  → Heavy (11B params) — only run once, on a subset, not across the full eval set
```

---

## The two CSR checks (locked definition — do not drift from this)

**Check 1 — Support (ALCE-style NLI):**
`NLI(premise=cited_chunk_text, hypothesis=atomic_claim)` → entailment = supported.

**Check 2 — Trace-grounding (faithfulness proxy):**
`cited_source_url ∈ set(source_urls retrieved across all tool calls in the logged trace)`.
Catches "phantom" citations to URLs the agent never saw.

**CSR formula:**

```
CSR = |{gold_source_urls that were retrieved AND end up cited on a
        Check-1-supported claim in the final answer}|
      ÷
      |{gold_source_urls that were retrieved}|
```

Three failure modes (use these labels in error analysis):

- **Dropped** — retrieved, gold-supporting, but not cited in the final answer at all.
- **Wrong-attributed** — info from that URL used, but a different URL was cited.
- **Phantom** — citation present in final answer, but the URL was never retrieved (caught by Check 2).

---

## Decisions still open (fill before Phase 2)

- [ ] Exact OpenRouter API endpoint / base URL confirmed working with `nvidia/nemotron-3-ultra-550b-a55b:free`
- [ ] Whether to run the T5 spot-check locally or via a hosted inference API (decision depends on available GPU)
