import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.config import get_settings
from hybrid_rag.pipeline import LLMUnavailable, Pipeline, RerankerUnavailable, Trace
from hybrid_rag.schema import Document

logger = logging.getLogger("hybrid_rag.query")


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)
    rerank: bool = Field(False, description="rescore the best candidates with the cross-encoder")


class Passage(BaseModel):
    doc_id: str
    title: str
    text: str


class TraceOut(BaseModel):
    stages_ms: dict[str, float]
    total_ms: float


class SearchResponse(BaseModel):
    passages: list[Passage]
    trace: TraceOut


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Passage]
    invalid_citations: list[int]
    retrieved: list[str]
    trace: TraceOut


def passage(document: Document) -> Passage:
    return Passage(doc_id=document.id, title=document.title, text=document.text)


def trace_out(trace: Trace) -> TraceOut:
    return TraceOut(stages_ms=trace.stages_ms, total_ms=trace.total_ms)


def get_pipeline(request: Request) -> Pipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="the pipeline is not loaded yet")
    return pipeline


PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """The API. Without `pipeline`, the real one is built when the server starts."""
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "pipeline", None) is None:
            from hybrid_rag.api.build import build_pipeline

            app.state.pipeline = build_pipeline(get_settings())
        yield

    app = FastAPI(title="hybrid-rag-eval", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(body: QueryRequest, pipeline: PipelineDep) -> SearchResponse:
        """Retrieval only: the best passages for the question. Needs no LLM."""
        trace = Trace()
        try:
            documents = pipeline.search(body.question, rerank=body.rerank, trace=trace)
        except RerankerUnavailable as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return SearchResponse(passages=[passage(d) for d in documents], trace=trace_out(trace))

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, pipeline: PipelineDep) -> QueryResponse:
        """Answer the question from the retrieved passages, with citations or a refusal."""
        try:
            answer, documents, trace = pipeline.ask(body.question, rerank=body.rerank)
        except RerankerUnavailable as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LLMUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502, detail=f"the LLM request failed ({type(error).__name__})"
            ) from error

        by_id = {d.id: d for d in documents}
        logger.info(
            json.dumps(
                {
                    "event": "query",
                    "question_chars": len(body.question),
                    "rerank": body.rerank,
                    "retrieved": [d.id for d in documents],
                    "refused": answer.refused,
                    "citations": answer.citations,
                    "invalid_citations": answer.invalid_citations,
                    "stages_ms": trace.stages_ms,
                    "total_ms": trace.total_ms,
                }
            )
        )
        return QueryResponse(
            answer=answer.text,
            refused=answer.refused,
            citations=[passage(by_id[doc_id]) for doc_id in answer.citations],
            invalid_citations=answer.invalid_citations,
            retrieved=[d.id for d in documents],
            trace=trace_out(trace),
        )

    return app


app = create_app()
