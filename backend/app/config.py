from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_transcription_model: str = "gpt-4o-transcribe"
    openai_note_model: str = "gpt-4.1-mini"
    database_path: Path = Path("data/consultations.db")
    audio_storage_dir: Path = Path("storage/audio")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
