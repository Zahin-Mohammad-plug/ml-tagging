"""Configuration for worker processes"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import os
from pydantic import field_validator, model_validator

class WorkerSettings(BaseSettings):
    """Worker-specific settings"""
    
    # Database - can be set directly via DATABASE_URL or constructed from POSTGRES_* vars
    database_url: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_db: Optional[str] = None
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    
    @model_validator(mode='after')
    def construct_database_url(self):
        """Construct database_url from POSTGRES_* env vars if not set directly"""
        if not self.database_url:
            # Read from environment variables (case-insensitive via pydantic-settings)
            user = self.postgres_user or os.getenv("POSTGRES_USER", "tagger")
            password = self.postgres_password or os.getenv("POSTGRES_PASSWORD", "tagger_dev_password")
            db = self.postgres_db or os.getenv("POSTGRES_DB", "ml_tagger")
            host = self.postgres_host or os.getenv("POSTGRES_HOST", "localhost")
            port_str = os.getenv("POSTGRES_PORT", "5432")
            port = self.postgres_port or int(port_str) if port_str else 5432
            self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        return self
    
    # Queue
    redis_url: str = "redis://:redis_dev_password@localhost:6379/0"
    
    # Worker Configuration
    worker_name: str = "default"
    worker_concurrency: int = 2
    log_level: str = "INFO"
    
    # Frame Sampling Configuration
    sample_fps: float = 0.5
    max_frames_per_scene: int = 200  # Increased from 100 to test if more frames improve accuracy
    sample_quality: str = "medium"
    cache_frames: bool = True
    
    # FFmpeg Configuration
    ffmpeg_threads: int = 2
    ffmpeg_timeout: int = 300
    
    # Vision Model Configuration
    vision_model: str = "clip-vit-base-patch32"
    vision_device: str = "auto"
    batch_size: int = 8
    embedding_batch_size: int = 4  # Reduced from 8 to prevent OOM - CLIP ViT-L/14 is memory intensive
    
    # ASR Configuration
    asr_model: str = "whisper-small"
    asr_device: str = "auto"
    asr_language: str = "en"
    
    # OCR Configuration
    ocr_engine: str = "paddleocr"
    ocr_languages: str = "en"  # Comma-separated language codes
    
    @property
    def ocr_languages_list(self) -> List[str]:
        """Parse OCR languages from comma-separated string"""
        if isinstance(self.ocr_languages, str):
            return [lang.strip() for lang in self.ocr_languages.split(',')]
        return self.ocr_languages
    
    # Audio Configuration
    audio_sample_rate: int = 16000
    audio_chunk_duration: int = 30
    
    # Fusion Configuration
    min_agreement_frames: int = 3
    temporal_window_seconds: int = 10
    
    # Signal Weights
    vision_weight: float = 0.7
    asr_weight: float = 0.2
    ocr_weight: float = 0.1
    
    # Thresholds
    default_review_threshold: float = 0.3
    default_auto_threshold: float = 0.8
    suggestion_min_score: float = 0.20  # Lowered from 0.3 to allow more tags through
    temporal_consistency_threshold: float = 0.25  # Raised from 0.20 to avoid false positives (require stronger matches)
    
    # Per-Tag Adaptive Thresholds
    # Brief tags: appear briefly but are accurate (need lower threshold to catch)
    tag_threshold_brief: float = 0.18
    # Common tags: standard threshold
    tag_threshold_common: float = 0.20
    # High-precision tags: prone to false positives (need higher threshold to reduce FPs)
    tag_threshold_high_precision: float = 0.29
    
    # Score Normalization
    # Disabled by default for better precision (reduces false positives)
    # Test results show no-normalization has 28% better precision and 25% fewer false positives
    enable_score_normalization: bool = False
    score_normalization_mode: str = "calibrate"  # Options: "preserve", "calibrate", "compress"
    debug_score_pipeline: bool = False  # Enable detailed score pipeline logging
    
    # Score Aggregation
    enable_aggregation_boosts: bool = True  # Enable temporal/peak boosts in Stage 7
    scoring_method: str = "mean"  # Options: "mean", "max", "max_frequency" - how to combine frame scores in Stage 3
    # "mean": Average all frame scores (good for common tags)
    # "max": Maximum frame score (good for brief tags)
    # "max_frequency": Max similarity * frequency with adaptive weighting (good for brief tags with some consistency)
    
    # Prompt Pooling Configuration
    prompt_pooling_method: str = "max"  # Options: "max", "softmax" - how to aggregate scores across multiple prompts
    # "max": Take maximum similarity across all prompts (original behavior)
    # "softmax": Use softmax pooling with temperature (smoother, less spiky scores)
    prompt_pooling_temperature: float = 0.07  # Temperature parameter for softmax pooling (lower = more selective)
    
    # CLIP Model Configuration
    clip_model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    
    # Calibration
    use_calibrated_confidence: bool = True
    calibration_model_path: str = "/models/calibration.pkl"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env that aren't in WorkerSettings

@lru_cache()
def get_worker_settings() -> WorkerSettings:
    """Get cached worker settings instance"""
    return WorkerSettings()