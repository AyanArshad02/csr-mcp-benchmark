# Citation Survival Rate (CSR): Do Citations Survive the MCP Tool Boundary?

**A benchmark measuring what happens to source attribution when a RAG knowledge base is consumed
by an AI agent over MCP instead of a single-shot retrieval call. It finds where citations are
lost and why, validates the metric against blind human judgement, and tests a guardrail that adds
missing citations back.**

> **Headline.** With the same model (gpt-4o-mini), corpus, questions and system prompt, an agent
> built with the **OpenAI Agents SDK** keeps far fewer of the sources it retrieved:
> CSR **0.323** [0.237, 0.414] vs **0.618** [0.528, 0.708] for single-shot RAG.
> 66 of its 127 answers cite nothing at all. A **LangGraph** agent shows **no** loss (0.614).
>
> **Cause.** It is not the agent loop and not sampling temperature. It is the API the SDK uses by
> default: switching the same agent from OpenAI's Responses API to Chat Completions cuts uncited
> answers from 54 to 18 and raises CSR by +0.108 (real), close to the single-shot baseline.
>
> **Fix.** Kapa's published system prompt for agents raises CSR by +0.111 (real) and leaves the SDK
> no longer significantly below baseline. Their recommended tool description alone does nothing.
>
> **Robustness.** The headline holds under three independent scorers (−0.31, −0.30, −0.30, all
> significant), and the metric agrees with a blind human check on 25–27 of 30 cases.

![CSR by setup, gpt-4o-mini, with 95% confidence intervals under the original scorer and an independent gpt-4o judge](results/figures/csr_by_setup.svg)

---

## The problem

Retrieval-augmented generation systems are judged on whether their *answers* are correct. But a
documentation assistant that is right and uncited is not much use: you cannot verify it, and you
cannot click through to the source.

The Model Context Protocol (MCP) makes this concrete. An MCP retrieval server hands
`{source_url, content}` chunks to an agent it does not control, and the agent writes the final
answer. The server's accuracy guarantees stop at that boundary. **What happens to attribution on
the other side of it is unmeasured.** This project measures it.

---

## What CSR measures

For every atomic claim in an answer, with its cited `source_url`s, two checks:

1. **Support check (ALCE-style):** does the cited chunk *support* the claim?
2. **Trace-grounding check:** was the cited URL *actually retrieved* by the agent, according to
   the logged tool-call trace?

> **CSR** = of the gold source URLs the agent **actually retrieved**, the fraction that end up in
> the final answer cited on a claim the chunk genuinely supports.

Conditioning on what was retrieved isolates *attribution* failure from *retrieval* failure. CSR is
`None` (not zero) when the agent retrieved no gold URL.

The support check is run three ways: the original scorer (`cross-encoder/nli-deberta-v3-base`,
argmax = entailment) and two LLM judges chosen against human labels (gpt-4o and Claude Sonnet 4.6,
same rubric as the human labellers). See [Is CSR trustworthy?](#is-csr-trustworthy).

---

## Results

127 human-verified FastAPI documentation questions (82 single-source, 29 multi-source,
16 unanswerable). Bootstrap 95% CIs, 10,000 resamples.

| Pipeline | Model | CSR (original scorer) | 95% CI | CSR (gpt-4o judge) | CSR (Claude judge) |
|---|---|---|---|---|---|
| Single-shot RAG | gpt-4o-mini | 0.618 | [0.528, 0.708] | 0.665 | 0.651 |
| LangGraph agent | gpt-4o-mini | 0.614 | [0.524, 0.705] | 0.700 | 0.652 |
| **OpenAI Agents SDK** | gpt-4o-mini | **0.323** | [0.237, 0.414] | **0.379** | **0.364** |
| Single-shot RAG | claude-sonnet-4-6 | 0.774 | [0.698, 0.844] | 0.887 | 0.835 |
| LangGraph agent | claude-sonnet-4-6 | 0.710 | [0.631, 0.790] | 0.804 | 0.766 |

The OpenAI Agents SDK × Claude combination was not run.

### Which differences are real

Every pipeline answers the *same* questions, so the test resamples the **per-question
difference**. Each comparison under all three scorers:

| Comparison | Original scorer | gpt-4o judge | Claude judge | Verdict |
|---|---|---|---|---|
| OpenAI Agents SDK − single-shot | **−0.311** [−0.434, −0.189] | **−0.296** [−0.413, −0.179] | **−0.296** [−0.418, −0.173] | **Real under all three** |
| LangGraph − single-shot (gpt-4o-mini) | +0.000 [−0.093, +0.093] | +0.029 [−0.044, +0.108] | +0.005 [−0.078, +0.088] | **No difference** |
| LangGraph − single-shot (Claude) | −0.077 [−0.168, +0.014] | −0.096 [−0.178, −0.019] | −0.082 [−0.173, +0.005] | Not established (1 of 3) |

"Agents lose citations" is **not** what the data says. One widely used SDK loses about half; a
LangGraph agent on the same model loses none.

---

## Why the OpenAI Agents SDK loses citations

A pre-registered ablation (plan and success criterion written in the decision log before running):
the same SDK agent, 127 questions each, **one change at a time**.

| Run | CSR (original) | CSR (gpt-4o judge) | Uncited answers | Paired change |
|---|---|---|---|---|
| Defaults, re-run | 0.333 | n/a | 60/127 | vs published run: −0.011 (replicates) |
| + temperature 0 | 0.412 | 0.431 | 54/127 | +0.082 (n.s.) |
| + Kapa's recommended tool description | 0.382 | n/a | 54/127 | −0.040 (n.s.) |
| **+ Chat Completions API** | **0.533** | **0.643** | **18/127** | **+0.108 / +0.201 (real)** |
| **+ Kapa's recommended system prompt** | **0.515** | **0.569** | 43/127 | **+0.111 / +0.136 (real)** |
| *Single-shot RAG, for reference* | *0.618* | *0.665* | *19/127* | |

Paired changes are vs the temperature-0 run (original scorer / gpt-4o judge).

- **Temperature is not the cause.** The original SDK pipeline left temperature at the API default
  while LangGraph used 0. Matching it helps a little but not significantly; the SDK remains clearly
  below single-shot (−0.207, real).
- **The API is.** The SDK calls OpenAI's Responses API by default. On Chat Completions (what the
  LangGraph agent uses), uncited answers fall to the single-shot level.
- **The SDK does not drop the sources in transport.** Logging exactly what the model receives
  shows every chunk arrives with its `source_url` intact; citations are lost in generation.
- **Kapa's own guidance helps when followed in full.** Their system prompt
  ([Prompt for grounded answers](https://docs.kapa.ai/retrieval/guides/prompt-your-agent))
  significantly improves CSR; their tool description alone does not, consistent with that page
  putting citation rules in the system prompt. Under that prompt the model often wrote plain
  `[title](url)` links instead of the requested `[[title](url)]` brackets; counting those too,
  uncited answers fall to 23/127.

Caveats: one run per setup; the Kapa run changes prompt wording and citation format together.
Files: `results/sdk_ablation_report.txt`, `results/rescore_compare_gpt-4o.txt`.

---

## Is CSR trustworthy?

Three checks, all blind (the labeller never saw a model's verdict before choosing their own):

1. **Which support checker agrees with humans?** 115 human judgements of "does this chunk support
   this sentence?" (`results/verifier_bench.txt`). On the 65-item held-out set: original DeBERTa
   scorer κ 0.24 on raw text; MiniCheck-Flan-T5 on cleaned prose κ 0.63 (0.67 with a code-symbol
   check); Claude judge κ 0.63;
   **gpt-4o judge κ 0.86** (precision 0.95, recall 0.95). Cleaning Markdown and code out of the
   input helps every model. Caveat: every pair was one DeBERTa had accepted when sampled, so this
   measures how well each checker rejects DeBERTa's false positives.
2. **Does the headline depend on the checker?** No. Re-scored with both LLM judges, the SDK drop is
   −0.30 under each, significant (table above). The judges find more support than DeBERTa in every
   pipeline (+0.03 to +0.11), so absolute CSR levels depend on the scorer; the comparisons do not.
3. **Does CSR agree with a human?** 30 gold sources, 6 per pipeline, judged "did this source
   survive, correctly cited?" (`results/csr_validation_report.txt`): agreement **25/30** (original
   scorer, κ 0.64), **27/30** (gpt-4o judge, κ 0.77), **26/30** (Claude judge, κ 0.70). Errors run
   mostly one way: scorers miss support a human sees, so CSR slightly under-counts survival.

---

## The guardrail

Post-processing middleware at the boundary a retrieval provider controls: it adds citations to
uncited sentences in a third-party agent's answer. **It never rewrites a word**:
`assert_only_citations_added()` checks that, after stripping citations and whitespace, the answer
is byte-identical, on every answer of every run.

| | v1 (naive) | v2 |
|---|---|---|
| Method | NLI on raw sentences vs every retrieved chunk; cite every hit | Prose only (code/Markdown stripped) → atomic claims → cleaned chunks → every claim entailed with P ≥ 0.9 (fixed before any held-out label) → cite the single best chunk |
| Citations added, 5 pipelines | 1,582 | 357 |
| CSR lift, OpenAI Agents SDK (original scorer) | +0.222 | +0.146 [+0.066, +0.227] |
| **Human-judged precision, blind** | **15/50 = 30%** | **39/50 = 78%** [65, 87] on a held-out sample |
| Precision on gold-source citations | 8/13 | 22/24 |

- **v1 fooled its own scorer.** The same NLI model inserted citations and graded them, rating its
  mostly-wrong insertions as precise; only the blind audit exposed it. Cause: raw Markdown (code
  blocks, `2. **Title**:` headings) and HTML-laden chunks produce confident false entailments
  (removing one bold heading moved P(entailment) from 0.997 to 0.000).
- **v2** was evaluated on a fresh sample committed before labelling (50 inserted + 15 it
  rejected, mixed blind): of what it rejected, only 4/15 were correct, so the threshold rejects
  the right things. A second rater (Claude, independently) agreed with the human on 54/65 (κ 0.68).
- v2 still misses the ≥ 90% precision bar; the next version moves citation support to the tool
  contract (short source IDs returned by the server) and uses an independent verifier.

Run it: `uv run python -m guardrail.run_guardrail --guardrail v2 --input … --output …`
(`--guardrail v1` is the default and reproduces the v1 numbers).

---

## Using this against Kapa's hosted MCP server

The local server mirrors Kapa's `search_<product>_knowledge_sources` contract (`{source_url,
content}`, Markdown content, streamable HTTP, `top_k`/`max_chars` defaults). **Pointing the
pipelines at a real Kapa project is not yet a one-line change**, and nothing here has been run
against Kapa's production system. What it needs:

1. **Server URL:** the pipelines currently hard-code `http://127.0.0.1:8000/mcp`
   (`pipelines/*.py`, `MCP_URL`); only the OpenAI SDK pipeline takes `--mcp-url`.
2. **Authentication:** Kapa's hosted server requires credentials (see
   [Hosted MCP server](https://docs.kapa.ai/retrieval/hosted-mcp-server)); no pipeline sends an
   auth header yet.
3. **Tool name:** the single-shot baseline calls `search_fastapi_knowledge_sources` by name; a
   Kapa project exposes `search_<your product>_knowledge_sources`.
4. **Eval set:** the 127 questions and gold URLs are FastAPI-specific. A real project needs its own
   hand-verified set; the metric, scorers and harness carry over unchanged.

---

## Reproducing

```bash
git clone <repo-url> && cd csr-mcp-benchmark
uv sync                                   # Python 3.12+, all versions pinned
cp .env.example .env                      # add OPENAI_API_KEY; AWS creds for the Claude arms

uv run pytest guardrail/ measure/ -q      # 35 tests, no API keys needed

# Rebuild the original-scorer numbers from the committed raw model outputs:
bash scripts/overnight_rebuild.sh
```

| What | Command |
|---|---|
| Guardrail v2 on all pipelines, v1 vs v2 table | `bash scripts/run_guardrail_v2_all.sh` then `uv run python -m measure.guardrail_v2_compare` |
| SDK ablation (generates new answers, ~$5 of API calls) | `bash scripts/run_sdk_ablation.sh` then `uv run python -m measure.sdk_ablation_report` |
| Verifier benchmark vs human labels | `uv run python -m measure.verifier_bench` |
| Re-score with an LLM judge | `uv run python -m measure.rescore_judge --judge gpt-4o` (or `claude`), then `uv run python -m measure.rescore_compare --judge gpt-4o` |
| CSR vs human | `uv run python -m measure.csr_validation_report` |
| Chart | `uv run python -m measure.make_charts` |

Every number in this README is reproducible from `results/`. LLM and NLI outputs are cached by
SHA-256 of their inputs, so re-scoring is deterministic and cheap once the cache is warm.

---

## How it is built

| Layer | Choice | Why |
|---|---|---|
| Corpus | FastAPI docs, raw Markdown from GitHub | Clean Markdown, stable public URLs, well-bounded |
| Chunking | Heading-aware recursive, with parent-heading breadcrumbs (2,085 chunks from 153 pages) | Each chunk stays a coherent, citable unit |
| Embeddings | `text-embedding-3-small`, persisted to Chroma | Simple, consistent with the OpenAI provider |
| Retrieval server | FastMCP, streamable HTTP | Mirrors a hosted MCP `search_<product>_knowledge_sources` contract |
| Pipelines | Single-shot · LangGraph ReAct · OpenAI Agents SDK | Same system prompt across pipelines |
| Original scorer | `cross-encoder/nli-deberta-v3-base`, argmax = entailment | Fast and local; kept as the reference |
| LLM judges | gpt-4o, Claude Sonnet 4.6, one fixed rubric, temperature 0 | Chosen against human labels; used as independent scorers |
| Claim decomposition | gpt-4o-mini, temperature 0, cached | FActScore-style; identical for every pipeline and scorer |

All dependency versions are pinned and `uv.lock` is committed.

---

## Honest limitations

- **One corpus, small sample.** 127 FastAPI questions (29 multi-source), each gold attribution
  verified by hand. Every number carries a confidence interval, and nulls are stated as nulls.
- **One run per setup.** Run-to-run variance is not measured, except that the SDK default
  replicated (0.323 → 0.333).
- **Two models, partial matrix.** gpt-4o-mini and Claude Sonnet 4.6; the OpenAI SDK × Claude
  combination was not run.
- **Absolute CSR depends on the scorer.** The LLM judges score 0.03–0.11 higher than the original
  scorer; conclusions are drawn from comparisons, which hold under all three.
- **Judges and models share families.** The gpt-4o judge scores gpt-4o-mini answers, the Claude
  judge scores Claude answers; each is cross-checked by the other family, and the Claude judge
  scores the Claude pipelines *lower* than the gpt-4o judge does.
- **Human checks are small.** 115 citation labels and 30 CSR units, mostly from one annotator,
  with an independent second rater (Claude) on 115 of them.
- **Self-hosted mirror.** This measures a local server implementing a published MCP contract, not
  Kapa's or anyone's production system.
- **NLI context window.** The original scorer truncates at 512 tokens; this likely explains part of
  why it finds less support than the judges.

---

## Engineering notes

The project was audited end to end before publication, as a hostile reviewer would. That found
**five bugs**, four of which would otherwise have shipped: a guardrail sentence splitter that cited
bare Markdown list markers (21% of insertions); a fabricated-citation counter that was 3–7× too
high; `ChatBedrock` silently defaulting to `max_tokens=1024`, truncating 52/127 answers in one arm;
an audit sampler that reverse-engineered decisions instead of recording them; and silent NLI
premise truncation. One of our own hypotheses was disproved: fixing the truncation did not erase
the Claude LangGraph drop (0.718 → 0.710).

Later checks found two more parity issues and are reported above rather than hidden: the guardrail
v1 circularity (exposed by a blind human audit) and an unset temperature in the SDK pipeline
(tested directly; not the cause).

---

## Repository layout

```
ingest/      corpus download, heading-aware chunking, embedding, Chroma index
server/      MCP retrieval server (streamable HTTP); opt-in Kapa tool description for ablations
eval_set/    127 human-verified Q&A pairs with gold source URLs
pipelines/   baseline · langgraph_agent · openai_agent (+ provider strategy layer, ablation flags)
measure/     scorer, LLM-judge re-scoring, verifier benchmark, CIs, paired tests, reports, chart
guardrail/   attribution middleware v1 and v2 + tests
results/     every published number, per-question traces, human labels, figures
scripts/     rebuild, guardrail v2 run, SDK ablation, labelling-page builder
```

---

## Background

Builds on **ALCE** (citation precision/recall via entailment) and **FActScore** (atomic claim
decomposition). Takes the *concept* from Wallat et al., that a correct answer is not the same as a
faithful one, while substituting a cheap trace-grounding check for their counterfactual
regeneration method. MiniCheck (Tang et al., 2024) is re-implemented from its official code for
the verifier benchmark.
