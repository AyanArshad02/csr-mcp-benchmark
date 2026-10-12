# Citation Survival Rate (CSR): Do Citations Survive the MCP Tool Boundary?

**A benchmark measuring what happens to source attribution when a RAG knowledge base is consumed
by an AI agent instead of a single-shot retrieval call — plus a citation guardrail, and a blind
human audit of what that guardrail actually inserts.**

> **Headline:** An agent built with the OpenAI Agents SDK retrieved the correct documentation and
> then answered **without citing it** — CSR **0.323** [0.237, 0.414] vs **0.618** [0.528, 0.708]
> for single-shot retrieval on the same model, same corpus, and a system prompt that explicitly
> demands inline citations. **66 of 127 answers contained no citation at all.**
>
> The same model inside a LangGraph agent showed **no** loss (0.614). Citation survival depends on
> the agent framework.
>
> A naive NLI guardrail that re-attaches citations raised that arm's CSR to 0.545 (+0.222) by our
> own scorer, but a **blind human audit found only 15 of 50 of its insertions correct**. The
> scorer approved the guardrail's own mistakes. A stricter v2 (+0.146 [+0.066, +0.227]) is
> under a held-out human audit. Until that is done, treat no guardrail number as verified recovery.

---

## The problem

Retrieval-augmented generation systems are judged on whether their *answers* are correct. But a
documentation assistant that is right and uncited is not much use — you cannot verify it, and you
cannot click through to the source.

The Model Context Protocol (MCP) makes this concrete. An MCP retrieval server hands raw
`{source_url, content}` chunks to an agent it does not control. The server's accuracy guarantees
stop at that boundary. **What happens to attribution on the other side of it is unmeasured.**

This project measures it.

---

## What CSR measures

For every atomic claim in an answer, with its cited `source_url`s, two cheap deterministic checks:

1. **Support check (ALCE-style):** does the cited chunk *entail* the claim?
   NLI with premise = chunk, hypothesis = claim.
2. **Trace-grounding check:** was the cited URL *actually retrieved* by the agent, across all its
   tool calls, according to the logged trace?

> **CSR** = of the gold source URLs the agent **actually retrieved**, what fraction end up in the
> final answer both correctly cited **and** attached to a claim the chunk genuinely supports.

Conditioning on what was retrieved is deliberate: it isolates *attribution* failure from
*retrieval* failure. A pipeline is never penalised for a document it never found.

CSR is `None` — not zero — when the agent retrieved no gold URL. That question is ineligible, not
failed.

---

## Results

127 human-verified FastAPI documentation questions (82 single-source, 29 multi-source,
16 unanswerable). Bootstrap 95% CIs, 10,000 resamples.

| Pipeline | Model | CSR | 95% CI | Multi-source | + Guardrail v1* | + Guardrail v2* |
|---|---|---|---|---|---|---|
| Baseline (single-shot) | gpt-4o-mini | 0.618 | [0.528, 0.708] | 0.518 | 0.660 | 0.632 |
| LangGraph ReAct agent | gpt-4o-mini | 0.614 | [0.524, 0.705] | 0.362 | 0.671 | 0.662 |
| **OpenAI Agents SDK** | gpt-4o-mini | **0.323** | [0.237, 0.414] | 0.179 | 0.545 | 0.470 |
| Baseline (single-shot) | claude-sonnet-4-6 | 0.774 | [0.698, 0.844] | 0.750 | 0.783 | 0.774 |
| LangGraph ReAct agent | claude-sonnet-4-6 | 0.710 | [0.631, 0.790] | 0.621 | 0.724 | 0.720 |

\* Guardrail columns are **scorer** numbers, not verified recovery. See [The guardrail](#the-guardrail):
a human audit found most v1 insertions wrong, and v2's audit is pending. The OpenAI Agents SDK ×
Claude arm was not run.

### Which differences are real

Every pipeline answers the *same* questions, so the correct test resamples the **per-question
difference**, not the two means separately.

| Comparison | Δ CSR | 95% CI | Verdict |
|---|---|---|---|
| Baseline → OpenAI Agents SDK | **−0.311** | [−0.434, −0.189] | **Real** |
| Baseline → OpenAI Agents SDK (multi-source) | **−0.339** | [−0.571, −0.107] | **Real** |
| Guardrail v1 lift, OpenAI Agents SDK (scorer) | +0.222 | [+0.136, +0.308] | Scorer-real; human audit: 15/50 insertions correct |
| Guardrail v2 lift, OpenAI Agents SDK (scorer) | +0.146 | [+0.066, +0.227] | Scorer-real; held-out human audit pending |
| Baseline → LangGraph, gpt-4o-mini | +0.000 | [−0.093, +0.093] | **Null** |
| Baseline → LangGraph, claude | −0.077 | [−0.168, +0.014] | Not significant |

**The interesting result is the contrast, not a blanket claim.** "Agents lose citations" is not
supported by this data — the LangGraph arm on gpt-4o-mini shows a difference of *exactly zero*.
One widely-used SDK lost half its citations on identical inputs; another lost none. That makes
citation survival a **fixable property of the agent framework**, not an inherent cost of agentic
retrieval.

---

## The guardrail

Post-processing middleware that sits where a retrieval provider already has control (the tool
boundary) and adds citations for third-party agents it cannot instrument. **It never rewrites a
word.** That is enforced in code: `assert_only_citations_added()` strips all citations and
whitespace from both answers and requires the remainder to be byte-identical, on every answer of
every run.

### v1 (naive): fooled its own scorer

For each uncited sentence, run NLI against every chunk in the agent's retrieval trace and cite
every chunk that entails it. By the scorer, this lifts the OpenAI Agents SDK arm from 0.323 to
0.545. 85% of the newly-surviving gold URLs were never cited anywhere in the original answer, so
the lift is not just finer scoring units. But *new* is not *correct*.

**Blind human audit** (50 insertions, seeded random, LangGraph × gpt-4o-mini, labelled without
seeing the model's verdict): **15 correct, 4 partial, 31 wrong**. On the 13 insertions pointing to
a gold source (the only ones that can move CSR), 8 were correct, 2 partial and 3 wrong.
Files: `results/audit_sample_50_verified.jsonl` and `results/audit_sample_50_claude_report.txt`.

Why it fails: v1 feeds raw Markdown "sentences" (code blocks, list headings like
`2. **Create a Dependency…**:`) and noisy HTML-laden chunks to the NLI model, which then gives
confident "entailment" to unrelated pairs. Removing one bold list heading moved P(entailment) from
0.997 to 0.000. With 5–10 chunks per question and every hit cited, false positives compound.

The scorer could not detect this. It rates v1's citation precision *higher* than v2's, because the
same NLI model inserts and scores. That circularity is the main limitation of using this guardrail
as evidence of recovery, and the audit is what exposed it.

### v2: stricter, under held-out audit

`guardrail/reattach_v2.py`: prose only (code and Markdown syntax stripped) → decompose into atomic
claims → clean the chunks → every claim must be entailed with **P ≥ 0.9** (threshold fixed before
any held-out label) → cite only the single best chunk.

- On the 50 audited items (now a dev set): 47 citations instead of 305, **0 of the 31 wrong ones
  reproduced**, 7 of the 15 correct ones kept. That's a recall cost.
- All 5 arms: 1,582 → 357 citations added. OpenAI Agents SDK lift **+0.146 [+0.066, +0.227]**
  (`results/guardrail_v2_compare.txt`).
- **Held-out audit:** 65 fresh items (50 inserted, 15 rejected below threshold), pooled across
  arms, excluding every v1-audited pair: `results/audit_v2_heldout.jsonl`. The sample was
  committed **before** labelling. Verdicts are not in yet, so v2's insertion precision is
  **unknown**.

Run it: `uv run python -m guardrail.run_guardrail --guardrail v2 --input … --output …`
(`--guardrail v1` is the default and reproduces the v1 numbers).

---

## Reproducing

```bash
git clone <repo-url> && cd csr-mcp-benchmark
uv sync                                   # Python 3.12+, all versions pinned
cp .env.example .env                      # add OPENAI_API_KEY; AWS creds for the Claude arms

uv run pytest guardrail/ measure/ -q      # 35 tests, no API keys needed

# Rebuild every published number from the committed raw model outputs:
bash scripts/overnight_rebuild.sh
```

`scripts/overnight_rebuild.sh` rescores all five pipelines, re-runs guardrail v1, verifies
determinism, and regenerates the confidence intervals, lift decomposition, and audit sample. It
aborts before touching any file if the tests fail.

Guardrail v2 on all five arms: `bash scripts/run_guardrail_v2_all.sh`, then
`uv run python -m measure.guardrail_v2_compare`.

To regenerate raw model outputs from scratch (costs API calls), start the MCP server
(`uv run python -m server.main`) and run the pipelines in `pipelines/`.

Every number in this README is reproducible from `results/`.

---

## How it is built

| Layer | Choice | Why |
|---|---|---|
| Corpus | FastAPI docs, raw Markdown from GitHub | Clean Markdown, stable public URLs, well-bounded |
| Chunking | Heading-aware recursive (2,085 chunks) | Each chunk stays a coherent citable unit |
| Embeddings | `text-embedding-3-small` | 153 files → 2,085 chunks, persisted to Chroma |
| Retrieval server | FastMCP, streamable HTTP | Mirrors a hosted MCP `search_<product>_knowledge_sources` contract |
| Pipelines | Single-shot · LangGraph ReAct · OpenAI Agents SDK | Agent prompts are byte-identical |
| NLI | `cross-encoder/nli-deberta-v3-base`, argmax = entailment | Fast enough for thousands of pairs |
| Claim decomposition | gpt-4o-mini, temperature 0, cached | FActScore-style; applied identically everywhere |

All dependency versions are pinned and `uv.lock` is committed. LLM and NLI outputs are cached by
SHA-256 of their inputs, so re-scoring is deterministic and cheap.

---

## Honest limitations

- **Circularity, measured.** The same NLI model inserts citations in the guardrail and scores
  them. A blind human audit showed this matters: 15/50 v1 insertions were correct, even though
  the scorer reported a large lift. Guardrail numbers are therefore reported as scorer numbers
  only, pending v2's held-out audit. An independent verifier (MiniCheck) is the next step if v2
  is still imprecise.
- **CSR itself is not yet validated against humans.** The audits above check guardrail
  insertions, not the CSR metric. A human-agreement check of CSR is planned.
- **Sample size.** 127 questions, each gold attribution verified by hand. A small verified set
  beats a large unverified one for a metric whose credibility rests entirely on its gold labels —
  which is why every number carries a confidence interval and null results are stated as nulls.
- **NLI context window.** DeBERTa truncates at 512 tokens; 10.9% of concatenated multi-citation
  premises exceed it. Measured effect: the scorer is marginally *too harsh*, identically across
  all pipelines, so it cannot bias a comparison.
- **Self-hosted mirror.** This measures a local server implementing a published MCP contract, not
  any vendor's production system.
- **Not causal.** We measure *that* one SDK drops citations, not the mechanism inside it. One
  untested candidate: the Agents SDK uses the Responses API, which may deliver the (identical)
  system prompt differently from Chat Completions. An ablation isolating this is planned.
- **Single run, two models.** Every arm is one cached run (bootstrap CIs, no repeated seeds), on
  gpt-4o-mini and Claude Sonnet 4.6. Citation instructions sit in the system prompt only, not in
  the MCP tool description.

---

## Engineering notes

The results above are the *second* set of numbers this project produced. Before publication it was
audited end to end as a hostile reviewer would, which found **five bugs** — four of which would
otherwise have shipped:

- A guardrail sentence splitter that cited bare Markdown list markers (21% of all insertions), and
  which **inflated** the originally-reported lift
- A fabricated-citation counter incremented per atomic claim instead of per citation (3–7× too high)
- `ChatBedrock` silently defaulting to `max_tokens=1024` while the baseline arm used 2048,
  truncating 52/127 answers in one arm and making the two Claude arms incomparable
- An audit sampler that reverse-engineered the guardrail's output instead of recording its decisions
- Silent NLI premise truncation

One hypothesis of our own was disproved by the data: we expected fixing the `ChatBedrock`
truncation to erase the Claude baseline → LangGraph drop. It did not (0.718 → 0.710 after
regenerating the arm).

Robustness checks now on record: determinism verified exactly across all five arms; a matched
98-question subset (controlling for differing retrieval recall) changes no conclusion; truncation
sensitivity is −0.001.

---

## Repository layout

```
ingest/      corpus download, heading-aware chunking, embedding, Chroma index
server/      MCP retrieval server (streamable HTTP)
eval_set/    127 human-verified Q&A pairs with gold source URLs
pipelines/   baseline · langgraph_agent · openai_agent (+ provider strategy layer)
measure/     scorer, bootstrap CIs, paired tests, lift decomposition, audit sampler
guardrail/   attribution middleware + its tests
results/     every published number, per-question traces, human audit labels, run logs
scripts/     full rebuild, guardrail v2 run, dev checks
```

---

## Background

Builds on **ALCE** (citation precision/recall via NLI entailment) and **FActScore** (atomic claim
decomposition). Takes the *concept* from Wallat et al. — that correct is not the same as faithful
— while deliberately substituting a cheap, deterministic trace-grounding proxy for their
counterfactual regeneration method.
