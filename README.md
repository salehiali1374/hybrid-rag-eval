# hybrid-rag-eval

![ci](https://github.com/salehiali1374/hybrid-rag-eval/actions/workflows/ci.yml/badge.svg)

Hybrid retrieval (BM25 + dense embeddings), reranking and evaluation for retrieval-augmented generation on Persian text. The goal is to measure, stage by stage, what each technique buys on a Persian question-answering benchmark, and to serve the result behind an API. Every number below is produced by a script in this repository and stored in [results/](results/).

```
question ─┬─> BM25 (sparse) ─┐
          └─> dense (e5) ────┴─> RRF fusion ─> [cross-encoder rerank] ─> top-5 passages ─> LLM ─> answer + citations / refusal
```

**Status:** milestones 0 to 8 of [docs/design.md](docs/design.md) are done. This README is milestone 9. Read [Limitations](#limitations-and-disclosures) before quoting any number: most samples are small.

## Data and protocol

[PersianQA](https://github.com/sajjjadayobi/PersianQA) (Persian Wikipedia, SQuAD format) turned into a retrieval benchmark: the corpus is all 993 unique paragraphs of the train and test files, and each answerable question has exactly one relevant paragraph, so recall@k is the same as hit rate@k. Constants are tuned on the train split (6306 questions) and everything below is reported on the test split (651 answerable questions) unless stated otherwise. One test question is 0.15 percentage points.

## Results

### 1. Retrieval (test, 651 questions)

| Method | recall@1 | MRR@10 | recall@10 | recall@20 | cost per question (CPU) |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0.9094 | 0.9372 | 0.9816 | 0.9862 | 0.82 ms |
| Dense (multilingual-e5-small) | 0.9386 | 0.9580 | 0.9862 | 0.9892 | 4.56 ms encode + 0.502 ms search (all queries in one batch) |
| Hybrid, RRF k=1, dense weight 1.5 (tuned on train) | 0.9616 | 0.9765 | 0.9969 | 0.9969 | BM25 + dense + fusion; not timed separately |
| Hybrid, RRF k=60, weight 1 (paper default) | 0.9647 | 0.9748 | 0.9939 | 0.9969 | same |
| Hybrid (tuned) + cross-encoder rerank of the top 10 (bge-reranker-v2-m3) | 0.9862 | 0.9916 | 0.9969 | 0.9969 | 12.008 s for the rerank step |

- Fusing BM25 and dense beats either alone: recall@1 0.9094 (BM25) and 0.9386 (dense) become 0.9616.
- The RRF constants were chosen on train by MRR@10 over a grid of 28 combinations: k=1, weight 1.5 won with 0.9423. The paper default (k=60, weight 1) scored 0.9232 on train, and k=5 was a near tie (0.9417). On test the default is even slightly ahead on recall@1 (0.9647 against 0.9616), so tuning gave no clear win.
- Reranking lifts recall@1 from 0.9616 to 0.9862 (about 16 more questions with the right paragraph first) but costs 12.008 s per question on this CPU, which rules it out for interactive use here. It only reorders the top 10, so recall@10 cannot change: the ceiling 0.9969 means 2 questions whose paragraph is not in the hybrid top 20 at all.
- Rerank depth 10 was picked because of its cost, not from test results.

### 2. Answer generation with citations and refusal (`gemma3:27b-it-q8_0`)

20 random answerable and 20 unanswerable questions per run, 5 passages shown to the model, which must cite passage numbers and reply `NO_ANSWER` when the passages do not answer the question. "Unanswerable" questions have no answer in the corpus, so refusing is correct.

| Run | Answerable: refused | Answerable: answer cites the right paragraph | Answerable: no or invalid citation | Unanswerable: correctly refused | Time per question |
| --- | --- | --- | --- | --- | --- |
| test, prompt v1 | 0% | 100% | 0% | 9/20 (45%) | 2.69 s |
| test, prompt v2 | 0% | 95% | 0% | 11/20 (55%) | 2.57 s |
| train, prompt v2 | 5% | 100% | 0% | 11/20 (55%) | 2.49 s |

- Citations work: no answer on the test sample had a missing or invalid citation, and the cited paragraph was the right one in 100% (v1) and 95% (v2) of the answers.
- Refusal is the weak point: the model refused only about half of the unanswerable questions. A first look at the unrefused ones suggests that some of these questions are partly covered by the retrieved passage, so not every one is a pure hallucination, but the model also answered some that the passages do not cover.
- Prompt v2 (stricter: answer only if a passage states the answer explicitly) was written after reading v1's failures on this same test sample, so its test numbers are not clean evidence. The train run is a second look and gives the same refusal rate. With 20 questions per group the difference between v1 and v2 is within noise.
- This is a measurement of one model on a self-hosted endpoint, not a benchmark of RAG generation in general.

### 3. LLM judge for answer correctness

The judge sees the question, the dataset's reference answers and the system answer, and says CORRECT or INCORRECT. It was checked against 50 human labels (one marked "unsure" and left out).

| Check | Result |
| --- | --- |
| Real answers, judge against human labels | agreement 49/49 (accuracy 100%, Cohen's kappa 1.0) |
| Share of those answers the human marked correct | 48/49 (98%) |
| Synthetic wrong answers (the answer to a different question) | 25/25 rejected |

Only **1 of the 49 real answers was wrong**, so the real data shows that the judge accepts correct answers but proves almost nothing about catching wrong ones, and kappa 1.0 rests on one negative example. The synthetic negatives are easy by construction. Treat the judge as validated for obvious errors only. The judge is also the same model that wrote the answers, which can bias it towards its own output (not measured here).

## Serving

```bash
cp .env.example .env     # optional: fill in an OpenAI-compatible LLM endpoint
docker compose up        # API on http://127.0.0.1:8000, interactive docs at /docs
```

| Endpoint | What it does |
| --- | --- |
| `GET /health` | liveness |
| `POST /search` | retrieval only (no LLM): `{"question": "...", "rerank": false}` returns the top passages |
| `POST /query` | answer with citations or a refusal: `answer`, `refused`, `citations`, `invalid_citations`, `retrieved`, `trace` |

Every response carries a per-stage trace (`bm25`, `dense`, `fusion`, optionally `rerank`, `generation`, in milliseconds), and each query writes one structured log line with timings, retrieved ids and citations, but not the question text. The reranker is off by default (`HRE_LOAD_RERANKER=true` loads it). The LLM endpoint comes only from `HRE_LLM_BASE_URL`, `HRE_LLM_MODEL` and `HRE_LLM_API_KEY`; nothing about it is stored in the code.

CI runs lint, the tests and a retrieval quality gate: BM25 on the test split must stay above the thresholds in [eval_gate.json](eval_gate.json), which are the committed baseline minus 0.01.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only build
pip install -e ".[dev,dense]"
pytest -m "not network"

python -m hybrid_rag.experiments.bm25 --split test                      # BM25
python -m hybrid_rag.experiments.dense --split test                     # dense (downloads the model)
python -m hybrid_rag.experiments.hybrid --split train --tune            # RRF constants, train only
python -m hybrid_rag.experiments.hybrid --split test --rrf-k 1 --dense-weight 1.5
python -m hybrid_rag.experiments.rerank --split test --depth 10         # slow: about 2 hours on CPU
python -m hybrid_rag.eval.gate                                          # the CI gate
```

Generation and judge experiments need an OpenAI-compatible endpoint:

```bash
export HRE_LLM_BASE_URL=http://<host>:<port>/v1 HRE_LLM_MODEL=<model name>
python -m hybrid_rag.experiments.generate --split test --n 20 --tag _prompt-v2
python -m hybrid_rag.experiments.make_labeling_set --n 50               # then label data/labeling/label.html
python -m hybrid_rag.experiments.judge_agreement
python -m hybrid_rag.experiments.judge_negatives --n 25
```

## Limitations and disclosures

- **One relevant paragraph per question**, so recall@k equals hit rate@k and nothing here measures how well a system ranks several relevant documents. The corpus is small (993 paragraphs) and BM25 and dense are already close to the ceiling, so differences between methods are a few questions.
- **No confidence intervals.** Each retrieval number is a single run on 651 questions; the generation and judge samples are 20 to 50 items. Read differences of a few points as noise.
- **Test split reuse.** Constants were tuned on train only, but the generation prompt v2 was changed after looking at v1's test failures (see above), and the rerank depth was fixed for cost.
- **The judge is weakly validated** (one real wrong answer) and shares its model with the generator.
- **Unanswerable questions are not all clean.** Some are about topics a retrieved passage partly covers, so "should have refused" is sometimes debatable.
- **Docker.** The image was built and served `/health`, `/search` and the 503 for a missing LLM in a container, using locally cached data and models. The first-start path of `docker compose up` (downloading the dataset and models into the volumes) and `/query` from inside a container were not exercised.
- **Cost.** Reranking is accurate but slow on CPU. A GPU or a smaller cross-encoder would change that trade-off; neither was measured.
- Not done on purpose: chat UI, authentication, multi-tenancy, fine-tuning, other vector databases.

## Repository layout

```
src/hybrid_rag/
  data/         PersianQA loader        retrieval/   bm25, dense, fusion, rerank, embedding cache
  text/         Persian normalization   generation/  prompt, citation and refusal parsing, LLM client, judge
  eval/         metrics, agreement, CI gate   pipeline.py   the pipeline behind the API
  experiments/  one script per result   api/          FastAPI app and startup
results/        every reported number (JSON) and the human labels
docs/design.md  goal, data, milestones
```
