from pydantic import BaseModel


DEFAULT_HMS_FIELDS = (
    "chief_complaint",
    "history",
    "examination",
    "diagnosis",
    "investigations",
    "treatment",
    "follow_up",
)

ASSIST_FIELDS = (
    "suggested_doctor_questions",
    "recommended_tests",
    "clinical_disclosure",
)

NOTE_FIELDS = DEFAULT_HMS_FIELDS + ASSIST_FIELDS


class ProcessAudioResponse(BaseModel):
    transcript: str
    note: dict[str, str]
    consultation_id: int | None = None


class Consultation(BaseModel):
    id: int
    created_at: str
    audio_path: str
    transcript: str
    note: dict[str, str]
    generated_note_json: dict[str, str]
