import logging
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError

from . import ai
from .config import settings
from .database import create_consultation, get_consultation, init_db, list_consultations, update_consultation
from .schemas import ASSIST_FIELDS, DEFAULT_HMS_FIELDS
from .schemas import Consultation, ProcessAudioResponse

ALLOWED_EXTENSIONS = {".webm", ".mp3", ".wav", ".ogg", ".opus"}
TRANSCRIPTION_EXTENSION_MAP = {
    ".opus": ".ogg",
}
REQUIRED_HMS_FIELDS = ("chief_complaint", "history")
FIELD_KEY_PATTERN = re.compile(r"[^a-z0-9_]+")

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


def recording_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


async def save_audio_upload(audio: UploadFile, consultation_id: int | None = None) -> Path:
    extension = Path(audio.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported audio formats: webm, mp3, wav, ogg, opus")

    storage_dir = Path(settings.audio_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    saved_extension = TRANSCRIPTION_EXTENSION_MAP.get(extension, extension)
    if consultation_id is None:
        file_name = f"pending_recording_{recording_timestamp()}_{uuid4().hex[:8]}{saved_extension}"
    else:
        file_name = f"consultation_{consultation_id}_recording_{recording_timestamp()}{saved_extension}"
    audio_path = storage_dir / file_name

    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    audio_path.write_bytes(contents)
    return audio_path


def rename_recording_for_consultation(audio_path: Path, consultation_id: int) -> Path:
    renamed_path = audio_path.with_name(
        f"consultation_{consultation_id}_recording_{recording_timestamp()}{audio_path.suffix}"
    )
    if renamed_path.exists():
        renamed_path = audio_path.with_name(
            f"consultation_{consultation_id}_recording_{recording_timestamp()}_{uuid4().hex[:8]}{audio_path.suffix}"
        )
    audio_path.rename(renamed_path)
    return renamed_path


def normalize_field_key(value: str) -> str:
    key = FIELD_KEY_PATTERN.sub("_", value.strip().lower().replace(" ", "_")).strip("_")
    return key[:50]


def parse_hms_fields(hms_fields_json: str | None) -> list[str]:
    fields = list(DEFAULT_HMS_FIELDS)
    if hms_fields_json:
        try:
            parsed = json.loads(hms_fields_json)
            if isinstance(parsed, list):
                fields = []
                for item in parsed:
                    raw_key = item.get("key") if isinstance(item, dict) else item
                    if not isinstance(raw_key, str):
                        continue
                    key = normalize_field_key(raw_key)
                    if key and key not in ASSIST_FIELDS and key not in fields:
                        fields.append(key)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid hms_fields JSON")

    cleaned = []
    for required_field in REQUIRED_HMS_FIELDS:
        if required_field not in cleaned:
            cleaned.append(required_field)
    for field in fields:
        if field not in cleaned and field not in ASSIST_FIELDS:
            cleaned.append(field)
    return cleaned[:20]


@app.post("/api/process-audio", response_model=ProcessAudioResponse)
async def process_audio(
    audio: UploadFile = File(...),
    hms_fields: str | None = Form(default=None),
    doctor_profile: str | None = Form(default=None),
) -> ProcessAudioResponse:
    try:
        note_fields = parse_hms_fields(hms_fields)
        audio_path = await save_audio_upload(audio)
        transcript = ai.transcribe_audio(str(audio_path))
        note = ai.generate_note(transcript, hms_fields=note_fields, doctor_profile=doctor_profile)
        consultation = create_consultation(str(audio_path), transcript, note)
        renamed_audio_path = rename_recording_for_consultation(audio_path, consultation["id"])
        update_consultation(consultation["id"], str(renamed_audio_path), transcript, note)
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

    return ProcessAudioResponse(transcript=transcript, note=note, consultation_id=consultation["id"])


@app.post("/api/consultations/{consultation_id}/append-audio", response_model=ProcessAudioResponse)
async def append_audio(
    consultation_id: int,
    audio: UploadFile = File(...),
    hms_fields: str | None = Form(default=None),
    doctor_profile: str | None = Form(default=None),
) -> ProcessAudioResponse:
    existing = get_consultation(consultation_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Consultation not found")

    try:
        note_fields = parse_hms_fields(hms_fields)
        audio_path = await save_audio_upload(audio, consultation_id=consultation_id)
        new_transcript = ai.transcribe_audio(str(audio_path))
        transcript = f"{existing['transcript']}\n\nFollow-up:\n{new_transcript}".strip()
        note = ai.generate_note(transcript, hms_fields=note_fields, doctor_profile=doctor_profile)
        audio_paths = f"{existing['audio_path']};{audio_path}"
        update_consultation(consultation_id, audio_paths, transcript, note)
    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.exception("Runtime error while appending audio")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OpenAIError as exc:
        logger.exception("OpenAI error while appending audio")
        raise HTTPException(status_code=502, detail=f"OpenAI processing failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error while appending audio")
        raise HTTPException(status_code=500, detail="Failed to append consultation audio") from exc

    return ProcessAudioResponse(transcript=transcript, note=note, consultation_id=consultation_id)


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
