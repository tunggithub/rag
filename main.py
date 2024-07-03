from uvicorn import run
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.services import conversation_router, qa_router, \
                         tool_calling_router, tool_generation_router


def _get_app():
    app = FastAPI(docs_url="/docs")

    origins = [
    "http://localhost:5050",
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(conversation_router, tags=["RAG API"])
    app.include_router(qa_router, tags=["RAG API"])
    app.include_router(tool_calling_router, tags=["RAG API"])
    app.include_router(tool_generation_router, tags=["RAG API"])
    return app

app = _get_app()

if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=8000, reload=True)
