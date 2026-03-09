"""Frame sampling worker using FFmpeg"""

import os
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import structlog
from celery import Task
import cv2
import numpy as np
from PIL import Image
import uuid

from .celery_app import app
from .database import get_database_connection, run_async
from .config import get_worker_settings

logger = structlog.get_logger()

def convert_scene_path(scene_path: str) -> str:
    """
    Convert a host-side video path to the container-accessible path.

    Uses MEDIA_HOST_PREFIX and MEDIA_CONTAINER_PREFIX environment variables.
    For example, if:
      MEDIA_HOST_PREFIX=C:/Videos/
      MEDIA_CONTAINER_PREFIX=/media/
    then C:/Videos/cooking/ep1.mp4 -> /media/cooking/ep1.mp4

    If no prefix mapping is configured the path is returned as-is.
    """
    if not scene_path:
        return scene_path

    # Normalise Windows separators
    path = scene_path.replace('\\', '/')

    host_prefix = os.environ.get("MEDIA_HOST_PREFIX", "").replace('\\', '/')
    container_prefix = os.environ.get("MEDIA_CONTAINER_PREFIX", "/media/")

    if host_prefix and path.startswith(host_prefix):
        relative = path[len(host_prefix):]
        return f"{container_prefix.rstrip('/')}/{relative.lstrip('/')}"

    return path

class SamplerTask(Task):
    """Base task class for sampler worker"""
    
    _db = None
    _settings = None
    
    @classmethod
    def on_bound(cls, app):
        """Called when task is bound to app"""
        cls._settings = get_worker_settings()
        cls._db = get_database_connection()
    
    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_worker_settings()
        return self._settings
    
    @property
    def db(self):
        if self._db is None:
            self._db = get_database_connection()
        return self._db

    def _extract_frames_ffmpeg(
        self, 
        scene_path: str, 
        fps: float, 
        max_frames: int,
        sampling_interval: int,
        job_id: str,
        is_stream: bool = False,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract frames using FFmpeg (supports both file and HTTP stream input)"""
        
        frames_data = []
        temp_dir = Path(tempfile.gettempdir()) / f"ml_tagger_{job_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # FFmpeg command to extract frames
            output_pattern = str(temp_dir / "frame_%06d.jpg")
            cmd = [
                "ffmpeg",
            ]
            
            # Add headers for HTTP streaming if needed (must be before -i)
            if is_stream and api_key:
                # FFmpeg uses -headers option for custom headers
                # Format: "HeaderName: HeaderValue\r\n"
                headers_str = f"ApiKey: {api_key}\r\n"
                cmd.extend(["-headers", headers_str])
            
            # Add input and output options
            cmd.extend([
                "-i", scene_path,
                "-vf", f"fps={fps}",
                "-q:v", "2",  # High quality
                "-threads", str(self.settings.ffmpeg_threads),
                "-y",  # Overwrite output files
                output_pattern
            ])
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.ffmpeg_timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Process extracted frames
            frame_files = sorted(temp_dir.glob("frame_*.jpg"))
            
            for i, frame_file in enumerate(frame_files[::sampling_interval][:max_frames]):
                if i >= max_frames:
                    break
                    
                # Calculate timestamp
                frame_number = i * sampling_interval
                timestamp = frame_number / fps
                
                # Get frame dimensions
                width, height = self._get_frame_dimensions(frame_file)
                
                # Optionally cache frame file or generate thumbnail
                cached_path = None
                if self.settings.cache_frames:
                    cached_path = self._cache_frame(frame_file, job_id, frame_number)
                
                frames_data.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp,
                    "file_path": str(cached_path) if cached_path else None,
                    "width": width,
                    "height": height,
                    "temp_path": str(frame_file)  # For immediate processing
                })
            
            return frames_data
            
        finally:
            # Cleanup temporary files if not caching
            if not self.settings.cache_frames:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_frame_dimensions(self, frame_path: Path) -> Tuple[int, int]:
        """Get frame dimensions"""
        try:
            with Image.open(frame_path) as img:
                return img.size
        except Exception:
            # Fallback using OpenCV
            img = cv2.imread(str(frame_path))
            if img is not None:
                height, width = img.shape[:2]
                return width, height
            return 0, 0

    def _cache_frame(self, frame_path: Path, job_id: str, frame_number: int) -> Path:
        """Cache frame file for later use"""
        # Use /app/.cache instead of /tmp/cache to avoid volume mount permission issues
        cache_dir = Path("/app/.cache/frames") / job_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cached_path = cache_dir / f"frame_{frame_number:06d}.jpg"
        
        # Copy or move frame to cache
        import shutil
        shutil.copy2(frame_path, cached_path)
        
        return cached_path

    def _store_frame_metadata(self, job_id: str, scene_id: str, frame_data: List[Dict[str, Any]]) -> List[str]:
        """Store frame metadata in database"""
        
        frame_ids = []
        
        # Use asyncio to handle async database operations  
        import asyncio
        import asyncpg
        
        async def store_frames():
            # Use direct database connection to avoid pool conflicts
            db_url = self.settings.database_url
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            connection = await asyncpg.connect(db_url)
            
            frame_ids_async = []
            
            try:
                for frame_info in frame_data:
                    frame_id = str(uuid.uuid4())
                    
                    # Insert into database
                    query = """
                        INSERT INTO frame_samples (id, job_id, scene_id, frame_number, timestamp_seconds, file_path, width, height)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """
                    
                    await connection.execute(
                        query,
                        frame_id,
                        job_id, 
                        scene_id,
                        frame_info["frame_number"],
                        frame_info["timestamp_seconds"], 
                        frame_info.get("file_path"),
                        frame_info.get("width", 0),
                        frame_info.get("height", 0)
                    )
                    
                    frame_ids_async.append(frame_id)
                    
                    logger.debug(
                        "Stored frame metadata",
                        frame_id=frame_id,
                        job_id=job_id,
                        frame_number=frame_info["frame_number"],
                        timestamp=frame_info["timestamp_seconds"]
                    )
                    
            finally:
                await connection.close()
            
            return frame_ids_async
        
        # Run the async storage
        frame_ids = run_async(store_frames())
        return frame_ids

@app.task(bind=True, base=SamplerTask, name='app.sampler.extract_frames')
def extract_frames(self, job_id: str, scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract frames from a video using FFmpeg.

    Args:
        job_id: Job identifier
        scene_data: Must include 'id' and 'path' (filesystem path to the video)

    Returns:
        Dictionary with extraction results
    """

    logger.info("Starting frame extraction", job_id=job_id, scene_id=scene_data.get("id"))

    try:
        scene_id = scene_data.get("id")
        if not scene_id:
            raise ValueError("Video ID not provided")

        scene_path = scene_data.get("path")
        if not scene_path:
            raise ValueError("Video path not provided in video data")

        input_path = convert_scene_path(scene_path)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        logger.debug("Using filesystem access", path=input_path, job_id=job_id)

        # Get job options from metadata (if available)
        from .database import get_job_metadata
        job_metadata = run_async(get_job_metadata(job_id))

        # Use job-specific options if provided, otherwise use settings defaults
        fps = job_metadata.get("sample_fps") if job_metadata.get("sample_fps") is not None else self.settings.sample_fps
        max_frames = job_metadata.get("max_frames_per_scene") if job_metadata.get("max_frames_per_scene") is not None else self.settings.max_frames_per_scene

        # Calculate sampling parameters
        duration = scene_data.get("duration", 0)

        # Determine sampling interval
        total_possible_frames = int(duration * fps) if duration else max_frames
        sampling_interval = max(1, total_possible_frames // max_frames) if total_possible_frames > max_frames else 1

        # Extract frames using FFmpeg
        frame_data = self._extract_frames_ffmpeg(
            scene_path=input_path,
            fps=fps,
            max_frames=max_frames,
            sampling_interval=sampling_interval,
            job_id=job_id,
        )

        # Store frame metadata in database
        frame_ids = self._store_frame_metadata(job_id, scene_id, frame_data)

        logger.info(
            "Frame extraction completed",
            job_id=job_id,
            scene_id=scene_id,
            frames_extracted=len(frame_ids),
        )

        return {
            "success": True,
            "frame_count": len(frame_ids),
            "frame_ids": frame_ids,
            "sampling_params": {
                "fps": fps,
                "max_frames": max_frames,
                "sampling_interval": sampling_interval
            },
        }

    except Exception as e:
        logger.error("Frame extraction failed", job_id=job_id, error=str(e))
        return {"success": False, "error": str(e)}

@app.task(name='app.sampler.cleanup_job_frames')
def cleanup_job_frames(job_id: str) -> Dict[str, Any]:
    """Clean up temporary frames for a job"""
    
    try:
        # Remove cached frames
        cache_dir = Path("/tmp/cache/frames") / job_id
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        
        # Remove temp directory
        temp_dir = Path(tempfile.gettempdir()) / f"ml_tagger_{job_id}"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        logger.info("Cleaned up frames", job_id=job_id)
        return {"success": True}
        
    except Exception as e:
        logger.error("Frame cleanup failed", job_id=job_id, error=str(e))
        return {"success": False, "error": str(e)}

# Utility functions for frame analysis
def analyze_frame_quality(frame_path: str) -> Dict[str, float]:
    """Analyze frame quality metrics"""
    
    img = cv2.imread(frame_path)
    if img is None:
        return {"blur_score": 0.0, "brightness": 0.0, "contrast": 0.0}
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur detection using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness (mean intensity)
    brightness = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    return {
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "contrast": float(contrast)
    }

def detect_scene_changes(frame_paths: List[str], threshold: float = 0.3) -> List[int]:
    """Detect scene changes between frames"""
    
    scene_changes = []
    prev_hist = None
    
    for i, frame_path in enumerate(frame_paths):
        img = cv2.imread(frame_path)
        if img is None:
            continue
            
        # Calculate histogram
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            # Compare histograms
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            
            if correlation < (1.0 - threshold):
                scene_changes.append(i)
        
        prev_hist = hist
    
    return scene_changes