from pydantic import BaseModel


NOTE_FIELDS = (
    "chief_complaint",
    "history",
    "examination",
    "diagnosis",
    "investigations",
    "treatment",
    "follow_up",
)


class ClinicalNote(BaseModel):
    chief_complaint: str = ""
    history: str = ""
    examination: str = ""
    diagnosis: str = ""
    investigations: str = ""
    treatment: str = ""
    follow_up: str = ""


class ProcessAudioResponse(BaseModel):
    transcript: str
    note: ClinicalNote


class Consultation(BaseModel):
    id: int
    created_at: str
    audio_path: str
    transcript: str
    note: ClinicalNote
    generated_note_json: ClinicalNote
