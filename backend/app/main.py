import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError

from . import ai
from .config import settings
from .database import create_consultation, init_db, list_consultations
from .schemas import Consultation, ProcessAudioResponse

ALLOWED_EXTENSIONS = {".webm", ".mp3", ".wav", ".ogg", ".opus"}
TRANSCRIPTION_EXTENSION_MAP = {
    ".opus": ".ogg",
}

logger = logging.getLogger(__name__)

app = FastAPI(title="HMS AI Scribe POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    Path(settings.audio_storage_dir).mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process-audio", response_model=ProcessAudioResponse)
async def process_audio(audio: UploadFile = File(...)) -> ProcessAudioResponse:
    extension = Path(audio.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported audio formats: webm, mp3, wav, ogg, opus")

    storage_dir = Path(settings.audio_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    saved_extension = TRANSCRIPTION_EXTENSION_MAP.get(extension, extension)
    audio_path = storage_dir / f"{uuid4().hex}{saved_extension}"

    try:
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        audio_path.write_bytes(contents)

        transcript = ai.transcribe_audio(str(audio_path))
        note = ai.generate_note(transcript)
        create_consultation(str(audio_path), transcript, note.model_dump())
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("Runtime error while processing audio")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OpenAIError as exc:
        logger.exception("OpenAI error while processing audio")
        raise HTTPException(status_code=502, detail=f"OpenAI processing failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing audio")
        raise HTTPException(status_code=500, detail="Failed to process consultation audio") from exc

    return ProcessAudioResponse(transcript=transcript, note=note)


@app.get("/api/consultations", response_model=list[Consultation])
def consultations() -> list[dict]:
    return list_consultations()

'''
Frontend static file serving - in production, the frontend would typically be served separately (e.g. via nginx or a CDN), but for simplicity we can serve it directly from FastAPI if the built files are present.
'''
from pathlib import Path
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")