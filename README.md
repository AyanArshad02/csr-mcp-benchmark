# Citation Survival Rate (CSR): Do Citations Survive the MCP Tool Boundary?

**A benchmark measuring what happens to source attribution when a RAG knowledge base is consumed
by an AI agent instead of a single-shot retrieval call — plus a middleware that recovers most of
the loss.**

> **Headline:** An agent built with the OpenAI Agents SDK retrieved the correct documentation and
> then answered **without citing it** — CSR **0.323** [0.237, 0.414] vs **0.618** [0.528, 0.708]
> for single-shot retrieval on the same model, same corpus, and a system prompt that explicitly
> demands inline citations. **66 of 127 answers contained no citation at all.**
>
> A thin NLI guardrail at the tool boundary recovers **+0.222 CSR** (0.323 → 0.545) without
> changing a single word of the agent's answer.

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

| Pipeline | Model | CSR | 95% CI | Multi-source | + Guardrail |
|---|---|---|---|---|---|
| Baseline (single-shot) | gpt-4o-mini | 0.618 | [0.528, 0.708] | 0.518 | 0.660 |
| LangGraph ReAct agent | gpt-4o-mini | 0.614 | [0.524, 0.705] | 0.362 | 0.671 |
| **OpenAI Agents SDK** | gpt-4o-mini | **0.323** | [0.237, 0.414] | 0.179 | **0.545** |
| Baseline (single-shot) | claude-sonnet-4-6 | 0.774 | [0.698, 0.844] | 0.750 | 0.783 |
| LangGraph ReAct agent | claude-sonnet-4-6 | 0.710 | [0.631, 0.790] | 0.621 | 0.724 |

### Which differences are real

Every pipeline answers the *same* questions, so the correct test resamples the **per-question
difference**, not the two means separately.

| Comparison | Δ CSR | 95% CI | Verdict |
|---|---|---|---|
| Baseline → OpenAI Agents SDK | **−0.311** | [−0.434, −0.189] | **Real** |
| Baseline → OpenAI Agents SDK (multi-source) | **−0.339** | [−0.571, −0.107] | **Real** |
| Guardrail lift, OpenAI Agents SDK | **+0.222** | [+0.136, +0.308] | **Real** |
| Guardrail lift, multi-source | **+0.393** | [+0.250, +0.554] | **Real** |
| Baseline → LangGraph, gpt-4o-mini | +0.000 | [−0.093, +0.093] | **Null** |
| Baseline → LangGraph, claude | −0.077 | [−0.168, +0.014] | Not significant |

**The interesting result is the contrast, not a blanket claim.** "Agents lose citations" is not
supported by this data — the LangGraph arm on gpt-4o-mini shows a difference of *exactly zero*.
One widely-used SDK lost half its citations on identical inputs; another lost none. That makes
citation survival a **fixable property of the agent framework**, not an inherent cost of agentic
retrieval.

---

## The guardrail

Post-processing middleware that sits where a retrieval provider already has control — the tool
boundary — and cites third-party agents it cannot instrument.

For each uncited sentence, run NLI against every chunk in the agent's own retrieval trace; where
a chunk entails the sentence, splice in `[source_url]`. **It never rewrites a word.** That claim
is enforced in code, not prose: `assert_only_citations_added()` strips all citations and
whitespace from both answers and requires the remainder to be byte-identical, on every answer of
every run.

**Is the lift real, or an artifact of finer scoring units?** Decomposing every gold URL that flips
to "survived": **85% is genuine new attribution** — URLs never cited anywhere in the original
answer. Only 15% is re-granularization.

---

## Reproducing

```bash
git clone <repo-url> && cd csr-mcp-benchmark
uv sync                                   # Python 3.12+, all versions pinned
cp .env.example .env                      # add OPENAI_API_KEY; AWS creds for the Claude arms

uv run pytest guardrail/ measure/ -q      # 25 tests, no API keys needed

# Rebuild every published number from the committed raw model outputs:
bash scripts/overnight_rebuild.sh
```

`scripts/overnight_rebuild.sh` rescores all five pipelines, re-runs the guardrail, verifies
determinism, and regenerates the confidence intervals, lift decomposition, and audit sample. It
aborts before touching any file if the tests fail.

To regenerate raw model outputs from scratch (costs API calls), start the MCP server
(`uv run python -m server.main`) and run the pipelines in `pipelines/`.

Every number in this README is reproducible from `results/`.

---

## How it is built

| Layer | Choice | Why |
|---|---|---|
| Corpus | FastAPI docs, raw Markdown from GitHub | Clean Markdown, stable public URLs, well-bounded |
| Chunking | Heading-aware recursive (1983 chunks) | Each chunk stays a coherent citable unit |
| Embeddings | `text-embedding-3-small` | 153 files → 1983 chunks, persisted to Chroma |
| Retrieval server | FastMCP, streamable HTTP | Mirrors a hosted MCP `search_<product>_knowledge_sources` contract |
| Pipelines | Single-shot · LangGraph ReAct · OpenAI Agents SDK | Agent prompts are byte-identical |
| NLI | `cross-encoder/nli-deberta-v3-base`, argmax = entailment | Fast enough for thousands of pairs |
| Claim decomposition | gpt-4o-mini, temperature 0, cached | FActScore-style; applied identically everywhere |

All dependency versions are pinned and `uv.lock` is committed. LLM and NLI outputs are cached by
SHA-256 of their inputs, so re-scoring is deterministic and cheap.

---

## Honest limitations

- **Circularity.** The same NLI model both inserts citations in the guardrail and scores them.
  Bounded three ways — the two checks are not identical, 85% of the lift is URLs never previously
  cited, and a 50-sample manual audit is in progress — but only an independent verifier
  (MiniCheck-FT5) fully resolves it. Scoped as future work.
- **Sample size.** 127 questions, each gold attribution verified by hand. A small verified set
  beats a large unverified one for a metric whose credibility rests entirely on its gold labels —
  which is why every number carries a confidence interval and null results are stated as nulls.
- **NLI context window.** DeBERTa truncates at 512 tokens; 10.9% of concatenated multi-citation
  premises exceed it. Measured effect: the scorer is marginally *too harsh*, identically across
  all pipelines, so it cannot bias a comparison.
- **Self-hosted mirror.** This measures a local server implementing a published MCP contract, not
  any vendor's production system.
- **Not causal.** We measure *that* one SDK drops citations, not the mechanism inside it.

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

The full bug log with measured impact is in `NOTES.md`; the reviewer-facing writeup, including a
**hypothesis of our own that the data disproved**, is in `b1-notes/notes/audit-and-defense.md`.

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
results/     every published number, per-question traces, run logs
b1-notes/    design notes, findings log, audit & defense writeup
```

---

## Background

Builds on **ALCE** (citation precision/recall via NLI entailment) and **FActScore** (atomic claim
decomposition). Takes the *concept* from Wallat et al. — that correct is not the same as faithful
— while deliberately substituting a cheap, deterministic trace-grounding proxy for their
counterfactual regeneration method.
