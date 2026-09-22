import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag.document_loader import DocumentError
from app.rag.generation import LLMGenerationError
from app.rag.pipeline import RAGPipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EvidenceRAG")
rag_pipeline = RAGPipeline()
static_directory = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_directory), name="static")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def frontend():
    return FileResponse(static_directory / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "indexed_chunks": len(rag_pipeline.vector_store.chunks),
    }


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Supported files are PDF, TXT, and MD")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name
        return rag_pipeline.index_file(temporary_path, filename=filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (DocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=500, detail="Document indexing failed") from exc
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        return rag_pipeline.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
