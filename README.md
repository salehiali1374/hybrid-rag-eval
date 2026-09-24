# hybrid-rag-eval

Hybrid retrieval (BM25 + dense embeddings), reranking and evaluation for retrieval-augmented generation on Persian text.

**Status: work in progress.** See [docs/design.md](docs/design.md) for the plan. Results will be added here as each milestone is completed.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not network"
```

Dense retrieval (milestone 3) also needs PyTorch and sentence-transformers. Install the CPU-only PyTorch build first to avoid downloading CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,dense]"
```
