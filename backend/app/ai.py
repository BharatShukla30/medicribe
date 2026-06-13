import json

from openai import OpenAI

from .config import settings
from .schemas import NOTE_FIELDS, ClinicalNote


NOTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {field: {"type": "string"} for field in NOTE_FIELDS},
    "required": list(NOTE_FIELDS),
}


SYSTEM_PROMPT = """
You convert gynecology consultation dictation into professional HMS-ready clinical documentation.
Return only facts present in the transcript. Do not infer or invent clinical details.
If a section is not mentioned, return an empty string for that field.
Use concise, professional medical language.
"""


def get_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.openai_api_key)


def transcribe_audio(audio_path: str) -> str:
    client = get_client()
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=settings.openai_transcription_model,
            file=audio_file,
            response_format="text",
        )
    return str(transcription).strip()


def generate_note(transcript: str) -> ClinicalNote:
    client = get_client()
    response = client.chat.completions.create(
        model=settings.openai_note_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "hms_clinical_note",
                "schema": NOTE_SCHEMA,
                "strict": True,
            },
        },
    )

    try:
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    clean_note = {field: str(parsed.get(field, "") or "") for field in NOTE_FIELDS}
    return ClinicalNote(**clean_note)
