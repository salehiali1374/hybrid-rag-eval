from fastapi import FastAPI

app = FastAPI(title="hybrid-rag-eval")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
