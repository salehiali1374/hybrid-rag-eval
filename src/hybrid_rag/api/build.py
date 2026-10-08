"""Build the real pipeline (corpus, indexes, models) once, when the API starts."""

from hybrid_rag.config import Settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.generation.client import OpenAICompatibleClient
from hybrid_rag.pipeline import Pipeline
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.retrieval.dense import DenseIndex, Encoder
from hybrid_rag.retrieval.embedding_cache import load_or_compute
from hybrid_rag.retrieval.rerank import Reranker
from hybrid_rag.text.normalize import tokenize


def build_pipeline(settings: Settings) -> Pipeline:
    # heavy imports, only needed when the real pipeline is built
    from sentence_transformers import CrossEncoder, SentenceTransformer

    corpus = load_persianqa(settings.data_dir).corpus
    docs = list(corpus.values())

    encoder = Encoder(SentenceTransformer(settings.embedding_model, device="cpu"))
    embeddings = load_or_compute(
        settings.data_dir / "embeddings",
        settings.embedding_model,
        "passage",
        [doc.text for doc in docs],
        encoder.encode_passages,
    )
    reranker = (
        Reranker(CrossEncoder(settings.reranker_model, device="cpu", max_length=512))
        if settings.load_reranker
        else None
    )
    llm = (
        OpenAICompatibleClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
        if settings.llm_model
        else None
    )
    return Pipeline(
        corpus,
        BM25({doc.id: tokenize(doc.text) for doc in docs}),
        DenseIndex([doc.id for doc in docs], embeddings),
        encoder,
        llm=llm,
        reranker=reranker,
    )
