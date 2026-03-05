"""Configuration management using Pydantic settings"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Optional, Union
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Database
    database_url: str = "postgresql+asyncpg://tagger:tagger_dev_password@localhost:5432/ml_tagger"
    
    # Queue
    redis_url: str = "redis://:redis_dev_password@localhost:6379/0"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: str = "INFO"
    
    # Security
    api_secret_key: str = "dev_secret_key_change_in_production"
    cors_origins: Union[List[str], str] = "http://localhost:3000,http://localhost:3001,http://localhost:9999"
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    # Processing Configuration
    max_concurrent_jobs: int = 3
    job_timeout_seconds: int = 1800  # 30 minutes
    
    # ML Processing Settings
    sample_fps: float = 0.5
    max_frames_per_scene: int = 100
    min_agreement_frames: int = 3
    temporal_window_seconds: int = 10
    
    # Signal Weights
    vision_weight: float = 0.7
    asr_weight: float = 0.2
    ocr_weight: float = 0.1
    
    # Thresholds
    default_review_threshold: float = 0.3
    default_auto_threshold: float = 0.8
    
    # Model Configuration
    vision_model: str = "clip-vit-base-patch32"
    vision_device: str = "auto"
    asr_model: str = "whisper-small"
    asr_device: str = "auto"
    ocr_engine: str = "paddleocr"
    clip_model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"  # CLIP model for embeddings
    
    # Advanced Settings
    use_calibrated_confidence: bool = True
    calibration_model_path: str = "/models/calibration.pkl"
    cache_frames: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore'
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()