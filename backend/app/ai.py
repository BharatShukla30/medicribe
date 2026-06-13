import json

from openai import OpenAI

from .config import settings
from .schemas import ASSIST_FIELDS, DEFAULT_HMS_FIELDS


SYSTEM_PROMPT = """
You convert consultation dictation into professional HMS-ready clinical documentation.
Return only facts present in the transcript. Do not infer or invent clinical details.
If a section is not mentioned, return an empty string for that field.
Use concise, professional medical language.

Also provide clinician-assist content:
- suggested_doctor_questions: concise follow-up questions the doctor may consider asking based only on gaps or concerns in the transcript.
- recommended_tests: possible investigations/tests for the doctor to consider based only on the transcript. Do not present these as instructions to the patient.
- clinical_disclosure: always state that questions and tests are AI-generated suggestions for clinician review, not a diagnosis, prescription, or substitute for medical judgment.
"""


def build_note_schema(note_fields: list[str]) -> dict:
    all_fields = note_fields + list(ASSIST_FIELDS)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {field: {"type": "string"} for field in all_fields},
        "required": all_fields,
    }


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


def generate_note(
    transcript: str,
    hms_fields: list[str] | None = None,
    doctor_profile: str | None = None,
) -> dict[str, str]:
    client = get_client()
    note_fields = hms_fields or list(DEFAULT_HMS_FIELDS)
    note_schema = build_note_schema(note_fields)
    profile = (doctor_profile or "").strip()
    profile_context = f"\nDoctor profile/specialty context: {profile}" if profile else ""
    response = client.chat.completions.create(
        model=settings.openai_note_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": (
                    f"HMS output fields to populate: {', '.join(note_fields)}."
                    f"{profile_context}\n\nTranscript:\n{transcript}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "hms_clinical_note",
                "schema": note_schema,
                "strict": True,
            },
        },
    )

    try:
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = {}

    clean_note = {field: str(parsed.get(field, "") or "") for field in note_fields + list(ASSIST_FIELDS)}
    if not clean_note["clinical_disclosure"]:
        clean_note["clinical_disclosure"] = (
            "AI-generated suggestions for clinician review only. "
            "Not a diagnosis, prescription, or substitute for medical judgment."
        )
    return clean_note
