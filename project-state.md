# project-state.md

> **Living project tracker.** Keep this in the repo root. Update the "WHERE I AM RIGHT NOW"
> section every working session. This is the single source of truth for *what's done, what's next,
> and what "done" means*. CLAUDE.md holds rules + locked decisions; this file holds **state + the
> finish line**.

---

## 0. The finish line — what "DONE" means (read this first)

The project is complete when ALL of these are true. This is your definition of done; nothing
ambiguous, nothing open-ended.

- [ ] **Server:** self-hosted MCP server mirrors Kapa's `search_<product>_knowledge_sources`
      contract over the FastAPI docs corpus; passes the MCP Inspector test (clean chunks, real
      URLs, relevance-ordered).
- [ ] **Eval set:** ~150 *human-verified* Q&A pairs with gold `source_url`s; includes single-source,
      multi-source, and unanswerable items; mixed difficulty.
- [ ] **Scorer:** computes **both CSR checks** (support + trace-grounding — see §2) and ALCE
      citation recall/precision/F1; passes all 5 hand-built unit tests.
- [ ] **Pipelines:** baseline + LangGraph + OpenAI Agents SDK all run over the eval set, each
      logging the **full tool-call trace** (needed for the trace-grounding check).
- [ ] **Headline table:** citation-F1 + CSR across {3 pipelines} × {2 models}, with variance from
      ≥3 seeds.
- [ ] **Falsification gate passed:** either a clear, consistent drop (thesis holds) OR an honest
      documented alternative finding. No fudged numbers.
- [ ] **Human-validation subset:** a small (~30-item) hand-labeled set confirms CSR tracks human
      judgment (agreement/correlation reported). This converts CSR from "internal heuristic" to
      "credible metric."
- [ ] **Guardrail:** thin middleware runs inline on both agent pipelines; before/after lift table
      with honest per-answer latency overhead and a completeness proxy.
- [ ] **Paper:** ~6–10 pages; every number traces to `results/`; generous limitations section;
      money chart (CSR / citation-F1 across pipelines, ideally CSR vs. retrieval-call-count).
- [ ] **Repo:** public, clean, secret-free, reproducible from a fresh clone in ~10 min; README
      leads with the headline number + money chart + 3-command reproduce steps; "swap in real Kapa"
      section present.
- [ ] **Outreach:** short, specific, number-led email sent to the founders, linking repo + paper.

If life forces a cut: **minimum viable = Phases 0–2 + a short writeup + send.** The guardrail
(Phase 3) elevates "diagnosis" to "diagnosis + fix" — cut it last, only if forced.

---

## 1. WHERE I AM RIGHT NOW

> **Update this block every session.**

- **Current phase:** 0 — Setup & foundations. Not yet started.
- **Last completed:** Read ALCE + "Correctness is not Faithfulness" papers; wrote thinking logs;
  locked the **trace-proxy** definition of CSR's faithfulness check (see §2).
- **Next concrete action:** Phase 0.1 — set up pinned environment + repo skeleton, fill in
  `DECISIONS.md`.
- **Open blockers:** none. (No real Kapa access yet — building on a self-hosted mirror; swapping in
  real Kapa is a later env-var change.)
- **Decisions still to finalize before coding:** see §6 (a few small ones — answer them first).

---

## 2. ⚠️ THE CORRECTED CSR DEFINITION (most important section — read before building the scorer)

This supersedes any earlier wording in CLAUDE.md or the master plan that defined CSR using only a
support check. **CSR has TWO checks, both cheap and deterministic. There is NO perturbation /
counterfactual regeneration in this project** (that was a deviation absorbed from the Wallat paper's
*method*; we keep Wallat's *concept*, not its method).

For each atomic claim in a final answer, with its cited `source_url`(s):

**Check 1 — Support (pure ALCE).**
Does the cited chunk *entail* the claim? premise = cited chunk text, hypothesis = claim, run NLI.
This is ALCE citation recall/precision. Deterministic, one NLI call per claim.

**Check 2 — Trace-grounding (the cheap faithfulness proxy).**
Was the cited `source_url` *actually present in the chunks the agent retrieved across all its tool
calls* (from the logged trace)? This is the free, operationalizable part of Wallat's faithfulness
idea — a **necessary-condition proxy**, not a full causal test.

**CSR combines them:**

> Of the gold-supporting `source_url`s that the agent **actually retrieved** (across all tool
> calls), what fraction end up in the final answer **(a) correctly cited** (Check 2: real, retrieved
> URL) **AND (b) on a claim the cited chunk genuinely supports** (Check 1: NLI entailment)?

A source fails CSR in one of three ways (name these in error analysis):
- **Dropped** — info used, but cited to a different source / no source (blending, lost-in-the-middle).
- **Wrong-attributed** — citation drifted to the wrong retrieved URL.
- **Phantom** — citation to a URL that supports nothing / was never retrieved (post-rationalization's
  visible signature; caught by Check 2).

**Paper framing sentence (keep verbatim-ish):**
> "ALCE reveals citation *support*; Wallat reveals support ≠ faithful *reliance*; CSR measures that
> gap using a trace-grounded support proxy. A full causal-faithfulness test (counterfactual
> regeneration) is out of scope; CSR uses retrieval-trace availability as a cheaper
> necessary-condition proxy for faithfulness."

**Why this matters:** stating clearly what you're deliberately *not* doing (and why) is itself the
engineering-judgment signal Kapa wants. Don't reintroduce perturbation scope creep.

---

## 3. The phases at a glance

| Phase | Name | Output | Rough time |
|---|---|---|---|
| 0 | Setup & foundations | Env, repo skeleton, decisions locked | 2–4 evenings |
| 1 | Measurement spine | MCP server + eval set + **two-check** scorer | 1.5–2 weekends |
| 2 | Pipelines & benchmark | Headline comparison table + falsification gate | 1.5 weekends |
| 3 | Guardrail & lift | Before/after table | 1 weekend |
| 4 | Write the paper | The PDF/writeup | 1 weekend |
| 5 | Package & send | Public repo + outreach email | 0.5 weekend |

Full step-by-step detail lives in `b1-notes/B1-MASTER-PLAN.md`. This file tracks state; the master
plan holds the how.

---

## 4. Phase checklists (tick as you go)

### Phase 0 — Setup
- [ ] Pinned env (`uv`/venv); key libraries import.
- [ ] `.env` with all 3 provider keys; `.env` gitignored **before first commit**.
- [ ] Repo skeleton + initial git commit.
- [ ] `DECISIONS.md` filled in (mirror of CLAUDE.md's table).
- [ ] Re-read notes 01, 06, 07, 08.
- [ ] §6 open decisions answered.

### Phase 1 — Measurement spine
- [ ] Ingestion: FastAPI docs → chunks (recursive by heading) → embeddings → Chroma, each chunk
      carrying canonical `source_url`.
- [ ] MCP server: one tool, `{source_url, content}` output, honors `top_k`/`max_chars`, streamable
      HTTP, passes Inspector.
- [ ] Eval set: ~150 human-verified items (single/multi/unanswerable, mixed difficulty) → `eval.jsonl`.
- [ ] Scorer: decomposition → **Check 1 (NLI support)** + **Check 2 (trace-grounding)** → citation
      recall/precision/F1 + CSR.
- [ ] Scorer passes 5 hand-built unit cases (one each: perfect, missing-citation, wrong-source,
      over-citing, phantom/unsupported).
- [ ] Decomposition + NLI spot-checked on ~15 real technical pairs.

### Phase 2 — Pipelines & benchmark
- [ ] Strategy layer (OpenAI / Anthropic / OpenRouter) — each tested with a hello call.
- [ ] Baseline pipeline (single retrieve → generate → cite); genuinely strong prompt.
- [ ] LangGraph agent (Kapa's pattern); logs full trace.
- [ ] OpenAI Agents SDK agent (same server); logs full trace.
- [ ] 30-item smoke test → inspect the gap before scaling.
- [ ] Full run → `results/main_table.csv`; variance from ≥3 seeds.
- [ ] **Falsification gate** passed (thesis holds OR honest alternative).
- [ ] Human-validation subset: CSR vs. human agreement reported.

### Phase 3 — Guardrail
- [ ] Middleware: stable citation tokens + post-gen NLI verification + flag/strip; framework-agnostic.
- [ ] Before/after table: F1/CSR off vs. on, Δ latency, % flagged, completeness proxy.
- [ ] Decision gate: if latency > ~300–500 ms/answer → smaller NLI model or batch; if completeness
      drops sharply → reframe as a tunable strictness knob.

### Phase 4 — Paper
- [ ] Methods + Results written first; every number traces to `results/`.
- [ ] CSR defined precisely with the two-check structure + the "out of scope: perturbation" note.
- [ ] Limitations section (NLI false positives, single corpus, self-hosted-not-real-Kapa,
      trace-proxy-not-causal-faithfulness).
- [ ] Money chart clean and self-explanatory.
- [ ] TL;DR readable by a non-expert.

### Phase 5 — Package & send
- [ ] Repo public, secret-free, reproducible from fresh clone (test it).
- [ ] README: thesis + headline number + money chart + 3-command reproduce + "swap in real Kapa".
- [ ] Outreach email: short, number-led subject, one specific ask, links repo + paper.
- [ ] Sent. (Then: one polite follow-up after ~10 days if no reply.)

---

## 5. Cross-cutting habits (every session)
- Commit often, honest messages. Git history is part of the artifact.
- Cache LLM + NLI outputs from the start.
- Log raw outputs + full traces, not just scores.
- Keep `NOTES.md` of surprises (raw material for the paper *and* for the Notion thinking-log).
- Re-verify external APIs (LangGraph, OpenAI Agents SDK, MCP SDK, Kapa schema) against current docs.
- Two fatal mistakes to never make: (1) leak an API key, (2) fudge a number.

---

## 6. ⚠️ FINALIZE BEFORE CODING (answer these, then start Phase 0)

These are the small open choices not yet locked. Decide each, record in `DECISIONS.md`, then begin.

1. **Which exact two models under test?** (e.g., frontier = GPT-5.1 or Claude Sonnet 4.6; free =
   which specific OpenRouter model?) — pin specific model IDs for reproducibility.
2. **Which model for claim decomposition?** (a cheap/fast one, applied identically to all pipelines.)
3. **Local NLI compute:** can you run `cross-encoder/nli-deberta-v3-base` locally (CPU ok, GPU
   better)? If not, plan a hosted-inference fallback. (The T5-XXL spot-check model is large — decide
   if you'll run it at all, or skip the spot-check.)
4. **Repo public from day one, or private until send?** (Public-from-start = cleaner history;
   private-until-send = no early exposure. Either is fine — just decide.)
5. **Notion thinking-log:** confirm you'll keep it updated in real time alongside `NOTES.md` (the
   anti-"AI-slop" authenticity artifact).

---

## 7. Outreach sequencing reminder (decided earlier)
- **Build first, reach out after** with the finished artifact. Don't ask for access before you have
  a result.
- The reverse-engineering repo (github.com/AyanArshad02/kapa-inspired-rag-mcp) is supporting
  evidence in the eventual email, not a reason to email now.
- Optional parallel low-stakes access route: Kapa Open Source Program (support@kapa.ai) if the repo
  qualifies.
- Do NOT post the thesis as a public comment on the founder's launch posts — save it for the email.
