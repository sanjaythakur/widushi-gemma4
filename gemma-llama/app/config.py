"""Application settings loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration for the FastAPI service.

    Values are read from the environment and (when present) a `.env` file at
    the repo root. Names are lowercased; environment variables are matched
    case-insensitively (e.g. ``LLAMA_SERVER_URL`` -> ``llama_server_url``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llama_server_url: str = Field(
        default="http://llama:8080",
        description="Base URL of the llama.cpp server.",
    )
    model_config_path: str = Field(
        default="model_configs/gemma4-e2b.yaml",
        description="Path to the active model YAML config (relative to CWD).",
    )
    llm_timeout_seconds: float = Field(
        default=1800.0,
        description=(
            "HTTP timeout (seconds) for llama.cpp requests. Default 1800s (30 min) "
            "because multimodal prompts on Pi 5 routinely take 5-15 min to encode "
            "before the first token is decoded; lower at your own risk."
        ),
    )
    llm_retries: int = Field(
        default=3, ge=0, description="Retry attempts for failed llama.cpp requests."
    )
    log_level: str = Field(default="info", description="Python logging level.")

    # ------------------------------------------------------------------
    # Memory & context engine (Phase 1A: file-backed learner profiles)
    # ------------------------------------------------------------------

    learner_profiles_dir: str = Field(
        default="/learner_profiles",
        description=(
            "Directory holding `<learner_id>.md` profile files. Bind-mounted "
            "from `./learner_profiles` into the api container; read-only in "
            "Phase 1A, flips to read-write in Phase 3 (onboarding)."
        ),
    )

    # ------------------------------------------------------------------
    # Memory & context engine (Phase 1B: SQLite-backed session/episode/turn)
    # ------------------------------------------------------------------

    data_dir: str = Field(
        default="/data",
        description=(
            "Directory that holds the SQLite database (and any future "
            "persistent state). Bind-mounted from `./data` into the api "
            "container so the DB survives restarts."
        ),
    )
    sqlite_path: str = Field(
        default="/data/widushi.db",
        description=(
            "Path to the SQLite database file (WAL mode, owned by the api "
            "container). Lives under `data_dir` by default."
        ),
    )
    session_idle_timeout_seconds: int = Field(
        default=1800,
        ge=60,
        description=(
            "If the most recent open session for a learner had its last "
            "turn (or, if no turns yet, was opened) more than N seconds "
            "ago, it is auto-closed and a fresh session is opened on the "
            "next request. Default 30 min."
        ),
    )
    working_block_k_turns: int = Field(
        default=3,
        ge=0,
        description=(
            "Number of most-recent turns from the current episode to "
            "include in the working block (T0). PRD §3.3 starting point."
        ),
    )
    episodic_block_max_episodes: int = Field(
        default=3,
        ge=0,
        description=(
            "Max prior episodes from the same session to include in the "
            "episodic block (T1, raw-turn dump until Phase 2 adds LLM "
            "summarisation)."
        ),
    )
    episodic_block_turns_per_episode: int = Field(
        default=2,
        ge=0,
        description=(
            "Per-prior-episode turn tail length included in the episodic "
            "block. Kept small so we never burn the budget on transcripts."
        ),
    )

    # ------------------------------------------------------------------
    # Memory & context engine (Phase 2: episode lifecycle + LLM summariser)
    # ------------------------------------------------------------------

    episode_summary_max_tokens: int = Field(
        default=128,
        ge=16,
        le=512,
        description=(
            "Token cap for the per-episode summary Gemma call (PRD Phase 2). "
            "128 is plenty for a ~2-sentence wrap-up; bumping it past 256 "
            "starts eating the episodic-block budget in §3.3."
        ),
    )
    episode_summary_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for the episode + rolling-summary calls. "
            "Kept low so summaries stay deterministic across re-runs."
        ),
    )
    rolling_summary_every_n_closes: int = Field(
        default=3,
        ge=1,
        description=(
            "Re-summarise session.rolling_summary after every Nth episode "
            "close (PRD Phase 2). Counter is persisted under "
            "session.meta_json so the cadence survives restarts."
        ),
    )

    # ------------------------------------------------------------------
    # TTS (Piper)
    # ------------------------------------------------------------------

    tts_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for the TTS subsystem. When false, every endpoint "
            "ignores tts=true and the /tts/* routes return 503."
        ),
    )
    piper_binary: str = Field(
        default="/usr/local/bin/piper",
        description="Path to the Piper binary inside the api container.",
    )
    tts_voices_dir: str = Field(
        default="/voices",
        description="Directory holding the downloaded Piper .onnx voices.",
    )
    tts_output_dir: str = Field(
        default="/tmp/tts_outputs",
        description="Directory for cached, URL-served WAV files.",
    )
    tts_output_ttl_seconds: int = Field(
        default=600,
        ge=30,
        description="TTL (s) before a cached WAV is evicted from tts_output_dir.",
    )
    tts_default_voice: str = Field(
        default="warm-academic",
        description="Personality id used when callers omit `voice`.",
    )
    tts_voices_enabled: str = Field(
        default="warm-academic,friendly-casual,neutral-news,energetic-kid,calm-storyteller",
        description=(
            "Comma-separated list of personality ids to download at api "
            "container startup. Trim to shorten cold-start."
        ),
    )
    tts_max_sentence_chars: int = Field(
        default=400,
        ge=80,
        description="Hard-flush threshold for the streaming sentence buffer.",
    )
    tts_max_concurrency: int = Field(
        default=2,
        ge=1,
        description=(
            "Max simultaneous Piper subprocesses. Pi 5 has 4 cores; keeping "
            "this at 2 leaves headroom for llama.cpp."
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
