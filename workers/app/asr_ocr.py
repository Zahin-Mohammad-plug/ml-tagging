"""
ASR/OCR Worker - Extracts audio transcription and on-screen text

This worker processes video files to extract:
1. Audio transcription using Whisper (ASR - Automatic Speech Recognition)
2. On-screen text using OCR (Optical Character Recognition)
"""

import uuid
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
from celery import Task
import structlog

from .celery_app import app
from .database import get_database_connection, store_text_analysis, get_job_frames
from .config import get_worker_settings
from .ml.interfaces import create_asr_engine, create_ocr_engine

logger = structlog.get_logger(__name__)


class AsrOcrTask(Task):
    """Base task with shared ASR and OCR engines"""
    
    _asr_engine = None
    _ocr_engine = None
    _db = None
    _settings = None
    
    @classmethod
    def on_bound(cls, app):
        """Initialize engines when task is bound to app"""
        super().on_bound(app)
        cls._asr_engine = None  # Lazy load on first use
        cls._ocr_engine = None
    
    @property
    def settings(self):
        """Get worker settings (cached)"""
        if self._settings is None:
            self._settings = get_worker_settings()
        return self._settings
    
    @property
    def db(self):
        """Get database connection (cached)"""
        if self._db is None:
            self._db = get_database_connection()
        return self._db
    
    @property
    def asr_engine(self):
        """Get ASR engine (cached and shared across tasks)"""
        if self._asr_engine is None:
            logger.info("Loading Whisper ASR engine")
            
            # Get model config from settings
            engine_type = self.settings.asr_engine_type if hasattr(self.settings, 'asr_engine_type') else "whisper"
            model_name = self.settings.asr_model_name if hasattr(self.settings, 'asr_model_name') else "small"
            device = self.settings.device if hasattr(self.settings, 'device') else "auto"
            
            self._asr_engine = create_asr_engine(
                engine_type=engine_type,
                model_name=model_name,
                device=device
            )
            
            logger.info(
                "ASR engine loaded",
                engine_type=engine_type,
                model_name=model_name,
                device=device
            )
        
        return self._asr_engine
    
    @property
    def ocr_engine(self):
        """Get OCR engine (cached and shared across tasks)"""
        if self._ocr_engine is None:
            logger.info("Loading OCR engine")
            
            # Get model config from settings
            engine_type = self.settings.ocr_engine_type if hasattr(self.settings, 'ocr_engine_type') else "paddleocr"
            languages = self.settings.ocr_languages if hasattr(self.settings, 'ocr_languages') else ["en"]
            
            self._ocr_engine = create_ocr_engine(
                engine_type=engine_type,
                languages=languages
            )
            
            logger.info(
                "OCR engine loaded",
                engine_type=engine_type,
                languages=languages
            )
        
        return self._ocr_engine


@app.task(bind=True, base=AsrOcrTask, name='app.asr_ocr.process_audio_text')
def process_audio_text(self, job_id: str, scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process audio transcription and OCR text extraction
    
    Args:
        job_id: Unique job identifier
        scene_data: Scene information including video file path
    
    Returns:
        {
            "success": bool,
            "asr_segment_count": int,
            "ocr_text_count": int,
            "total_text_entries": int
        }
    """
    
    logger.info(
        "Starting ASR/OCR processing",
        job_id=job_id,
        scene_id=scene_data.get("id")
    )
    
    try:
        text_results = []
        
        # Process audio transcription (ASR)
        asr_results = _process_audio(self, job_id, scene_data)
        text_results.extend(asr_results)
        
        logger.info(
            "ASR processing completed",
            job_id=job_id,
            segment_count=len(asr_results)
        )
        
        # Process on-screen text (OCR)
        ocr_results = _process_ocr(self, job_id)
        text_results.extend(ocr_results)
        
        logger.info(
            "OCR processing completed",
            job_id=job_id,
            text_count=len(ocr_results)
        )
        
        # Store all text analysis results in database
        if text_results:
            from .database import run_async
            run_async(store_text_analysis(text_results))
        
        logger.info(
            "ASR/OCR processing completed",
            job_id=job_id,
            asr_segments=len(asr_results),
            ocr_texts=len(ocr_results),
            total_entries=len(text_results)
        )
        
        return {
            "success": True,
            "asr_segment_count": len(asr_results),
            "ocr_text_count": len(ocr_results),
            "total_text_entries": len(text_results)
        }
        
    except Exception as e:
        logger.error(
            "ASR/OCR processing failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}


def _process_audio(task_self, job_id: str, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and transcribe audio from video
    
    Args:
        job_id: Job identifier
        scene_data: Scene information with video path
    
    Returns:
        List of ASR result dictionaries
    """
    
    scene_path = scene_data.get("path")
    if not scene_path or not Path(scene_path).exists():
        logger.warning(
            "Video file not found for ASR",
            job_id=job_id,
            scene_path=scene_path
        )
        return []
    
    asr_results = []
    
    try:
        # Extract audio to temporary WAV file
        audio_path = _extract_audio(task_self, scene_path, job_id)
        
        if not audio_path:
            logger.warning("Audio extraction failed", job_id=job_id)
            return []
        
        # Transcribe audio using Whisper
        language = scene_data.get("audio_language")  # Can be None for auto-detection
        transcription = task_self.asr_engine.transcribe_audio(str(audio_path), language=language)
        
        # Store each segment as a text analysis entry
        for segment in transcription.get("segments", []):
            asr_results.append({
                "id": str(uuid.uuid4()),
                "frame_id": None,  # ASR not tied to specific frames
                "job_id": job_id,
                "analysis_type": "asr",
                "text_content": segment["text"],
                "confidence": segment.get("confidence", 0.0),
                "language": transcription.get("language", "unknown"),
                "start_time": segment.get("start"),
                "end_time": segment.get("end"),
                "bounding_box": None,
                "metadata": {
                    "model": task_self.settings.asr_model_name if hasattr(task_self.settings, 'asr_model_name') else "whisper-small",
                    "full_text": transcription.get("text", "")
                }
            })
        
        # Clean up temp audio file
        if audio_path.exists():
            audio_path.unlink()
        
        logger.info(
            "Audio transcription completed",
            job_id=job_id,
            segment_count=len(asr_results),
            language=transcription.get("language")
        )
        
    except Exception as e:
        logger.error(
            "Audio processing failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
    
    return asr_results


def _extract_audio(task_self, video_path: str, job_id: str) -> Optional[Path]:
    """
    Extract audio track from video using FFmpeg
    
    Args:
        video_path: Path to video file
        job_id: Job identifier for temp file naming
    
    Returns:
        Path to extracted audio file or None if failed
    """
    
    try:
        # Create temp file for audio
        temp_audio = Path(tempfile.gettempdir()) / f"ml_tagger_audio_{job_id}.wav"
        
        # FFmpeg command to extract audio as WAV
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate (Whisper standard)
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            str(temp_audio)
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.warning(
                "Audio extraction failed",
                job_id=job_id,
                error=result.stderr
            )
            return None
        
        if not temp_audio.exists():
            logger.warning("Audio file not created", job_id=job_id)
            return None
        
        logger.debug(
            "Audio extracted",
            job_id=job_id,
            audio_path=str(temp_audio),
            size_mb=temp_audio.stat().st_size / (1024 * 1024)
        )
        
        return temp_audio
        
    except Exception as e:
        logger.error(
            "Audio extraction error",
            job_id=job_id,
            error=str(e)
        )
        return None


def _process_ocr(task_self, job_id: str) -> List[Dict[str, Any]]:
    """
    Extract text from video frames using OCR
    
    Args:
        task_self: Task instance
        job_id: Job identifier
    
    Returns:
        List of OCR result dictionaries
    """
    
    ocr_results = []
    
    try:
        # Get frames from database
        frames = get_job_frames(job_id)
        
        if not frames:
            logger.warning("No frames found for OCR", job_id=job_id)
            return []
        
        # Sample frames for OCR (don't process all frames - too expensive)
        ocr_sample_rate = task_self.settings.ocr_sample_rate if hasattr(task_self.settings, 'ocr_sample_rate') else 5
        sampled_frames = frames[::ocr_sample_rate]
        
        logger.info(
            "Processing OCR on sampled frames",
            job_id=job_id,
            total_frames=len(frames),
            sampled_frames=len(sampled_frames)
        )
        
        # Process frames with OCR
        for frame in sampled_frames:
            frame_path = frame.get("file_path")
            
            if not frame_path or not Path(frame_path).exists():
                logger.warning(
                    "Frame file not found for OCR",
                    job_id=job_id,
                    frame_id=frame.get("id")
                )
                continue
            
            try:
                # Run OCR on frame
                ocr_result = task_self.ocr_engine.extract_text(frame_path)
                
                # Only store if text was found
                if ocr_result.get("text") and ocr_result.get("confidence", 0) > 0.5:
                    # Store each detected text box separately
                    for box in ocr_result.get("boxes", []):
                        ocr_results.append({
                            "id": str(uuid.uuid4()),
                            "frame_id": frame["id"],
                            "job_id": job_id,
                            "analysis_type": "ocr",
                            "text_content": box["text"],
                            "confidence": box["confidence"],
                            "language": None,  # OCR doesn't reliably detect language
                            "start_time": frame.get("timestamp_seconds"),
                            "end_time": frame.get("timestamp_seconds"),
                            "bounding_box": box.get("bbox"),
                            "metadata": {
                                "frame_number": frame.get("frame_number"),
                                "ocr_engine": task_self.settings.ocr_engine_type if hasattr(task_self.settings, 'ocr_engine_type') else "paddleocr"
                            }
                        })
                
                logger.debug(
                    "OCR processed frame",
                    job_id=job_id,
                    frame_id=frame["id"],
                    texts_found=len(ocr_result.get("boxes", []))
                )
                
            except Exception as e:
                logger.warning(
                    "OCR failed for frame",
                    job_id=job_id,
                    frame_id=frame.get("id"),
                    error=str(e)
                )
                continue
        
        logger.info(
            "OCR processing completed",
            job_id=job_id,
            texts_extracted=len(ocr_results)
        )
        
    except Exception as e:
        logger.error(
            "OCR processing error",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
    
    return ocr_results


@app.task(bind=True, base=AsrOcrTask, name='app.asr_ocr.process_audio_only')
def process_audio_only(self, job_id: str, scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process only audio transcription (skip OCR)
    
    This can be used for faster processing when OCR is not needed.
    
    Args:
        job_id: Unique job identifier
        scene_data: Scene information including video file path
    
    Returns:
        {
            "success": bool,
            "segment_count": int
        }
    """
    
    logger.info("Starting audio-only processing", job_id=job_id)
    
    try:
        asr_results = _process_audio(self, job_id, scene_data)
        
        if asr_results:
            from .database import run_async
            run_async(store_text_analysis(asr_results))
        
        logger.info(
            "Audio-only processing completed",
            job_id=job_id,
            segment_count=len(asr_results)
        )
        
        return {
            "success": True,
            "segment_count": len(asr_results)
        }
        
    except Exception as e:
        logger.error(
            "Audio-only processing failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}


@app.task(bind=True, base=AsrOcrTask, name='app.asr_ocr.process_ocr_only')
def process_ocr_only(self, job_id: str) -> Dict[str, Any]:
    """
    Process only OCR text extraction (skip ASR)
    
    This can be used when audio is not available or not needed.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        {
            "success": bool,
            "text_count": int
        }
    """
    
    logger.info("Starting OCR-only processing", job_id=job_id)
    
    try:
        ocr_results = _process_ocr(self, job_id)
        
        if ocr_results:
            from .database import run_async
            run_async(store_text_analysis(ocr_results))
        
        logger.info(
            "OCR-only processing completed",
            job_id=job_id,
            text_count=len(ocr_results)
        )
        
        return {
            "success": True,
            "text_count": len(ocr_results)
        }
        
    except Exception as e:
        logger.error(
            "OCR-only processing failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}
