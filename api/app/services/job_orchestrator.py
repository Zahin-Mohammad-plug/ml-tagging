"""Job orchestration service for coordinating ML processing pipeline"""

from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime, timedelta
import pytz
import structlog
from redis import Redis
from celery import Celery
import sqlalchemy as sa
from concurrent.futures import ThreadPoolExecutor

from ..database import async_session_maker, jobs_table, frame_samples_table, embeddings_table
from ..models import Job, JobStatus, JobPriority, VideoScene
from ..config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Thread pool for blocking Celery calls
executor = ThreadPoolExecutor(max_workers=10)

# Celery app for job queue
celery_app = Celery(
    "ml-tagger",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Configure task routing to match worker configuration
celery_app.conf.update(
    task_routes={
        'app.sampler.extract_frames': {'queue': 'sampling'},
        'app.embeddings.generate_embeddings': {'queue': 'embeddings'},
        'app.asr_ocr.process_audio_text': {'queue': 'asr_ocr'},
        'app.fusion.generate_suggestions': {'queue': 'fusion'},
    },
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


class JobOrchestrator:
    """Orchestrates the ML processing pipeline for scenes"""
    
    def __init__(self):
        self.redis = Redis.from_url(settings.redis_url)
        # Read MAX_CONCURRENT_JOBS from settings (environment variable)
        self.MAX_CONCURRENT_JOBS = settings.max_concurrent_jobs

    async def _execute_query(self, query, *, fetch: Optional[str] = None, commit: bool = False):
        """Execute a SQLAlchemy query using a fresh async session."""

        async with async_session_maker() as session:
            try:
                result = await session.execute(query)
                
                # Materialize results BEFORE commit/close to ensure all data is loaded
                if fetch == "one":
                    row = result.mappings().first()
                    if row:
                        # Convert RowMapping to dict immediately while session is open
                        row_dict = {key: value for key, value in row.items()}
                        if commit:
                            await session.commit()
                        return row_dict
                    else:
                        if commit:
                            await session.commit()
                        return None
                elif fetch == "all":
                    rows = result.mappings().all()
                    # Convert RowMapping objects to dicts immediately while session is open
                    rows_list = [{key: value for key, value in row.items()} for row in rows] if rows else []
                    if commit:
                        await session.commit()
                    return rows_list
                elif fetch == "scalar":
                    scalar_value = result.scalar_one_or_none()
                    if commit:
                        await session.commit()
                    return scalar_value
                else:
                    if commit:
                        await session.commit()
                    return None
            except Exception:
                await session.rollback()
                raise

    async def create_job(
        self,
        scene_id: str,
        priority: JobPriority = JobPriority.NORMAL,
        force_reprocess: bool = False,
        max_frames_per_scene: Optional[int] = None,
        sample_fps: Optional[float] = None,
        auto_approve_threshold: Optional[float] = None,
        auto_delete_threshold: Optional[float] = None,
    ) -> Job:
        """Create a new processing job."""

        if not force_reprocess:
            recent_job = await self._get_recent_job(scene_id)
            if recent_job and recent_job.status == JobStatus.COMPLETED and recent_job.completed_at:
                time_since = datetime.now(pytz.UTC) - recent_job.completed_at
                if time_since < timedelta(hours=24):
                    logger.info(
                        "Scene recently processed, skipping",
                        scene_id=scene_id,
                        last_processed=recent_job.completed_at,
                    )
                    return recent_job

        job_data = {
            "scene_id": scene_id,
            "status": JobStatus.QUEUED,
            "priority": priority,
            "progress": {
                "sampling": {"status": "pending"},
                "embeddings": {"status": "pending"},
                # "asr_ocr": {"status": "pending"},  # ASR/OCR disabled
                "fusion": {"status": "pending"},
            },
            "metadata": {
                "force_reprocess": force_reprocess,
                "created_by": "api",
                "max_frames_per_scene": max_frames_per_scene,
                "sample_fps": sample_fps,
                "auto_approve_threshold": auto_approve_threshold,
                "auto_delete_threshold": auto_delete_threshold,
            },
        }

        query = jobs_table.insert().values(**job_data).returning(jobs_table)
        job_record = await self._execute_query(query, fetch="one", commit=True)

        if not job_record:
            raise RuntimeError("Failed to create job record")

        job = Job(**job_record)
        logger.info("Job created", job_id=job.job_id, scene_id=scene_id, priority=priority)
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""

        query = sa.select(jobs_table).where(jobs_table.c.job_id == job_id)
        record = await self._execute_query(query, fetch="one")
        return Job(**record) if record else None

    async def get_active_job(self, scene_id: str) -> Optional[Job]:
        """Get active job for a scene (if any)."""

        query = (
            sa.select(jobs_table)
            .where(
                jobs_table.c.scene_id == scene_id,
                jobs_table.c.status.in_(
                    [
                        JobStatus.QUEUED,
                        JobStatus.SAMPLING,
                        JobStatus.EMBEDDINGS,
                        JobStatus.ASR_OCR,
                        JobStatus.FUSION,
                    ]
                ),
            )
            .order_by(jobs_table.c.created_at.desc())
        )

        record = await self._execute_query(query, fetch="one")
        return Job(**record) if record else None

    async def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[Job]:
        """List recent jobs, optionally filtered by status."""

        query = sa.select(jobs_table).order_by(jobs_table.c.created_at.desc()).limit(limit)

        if status:
            try:
                status_enum = JobStatus(status)
                query = query.where(jobs_table.c.status == status_enum)
            except ValueError:
                logger.warning("Invalid job status filter", status=status)

        records = await self._execute_query(query, fetch="all")
        return [Job(**record) for record in (records or [])]

    async def count_active_processing_jobs(self) -> int:
        """Count jobs currently in processing states (not queued or completed)."""
        query = sa.select(sa.func.count(jobs_table.c.job_id)).where(
            jobs_table.c.status.in_(
                [
                    JobStatus.SAMPLING,
                    JobStatus.EMBEDDINGS,
                    JobStatus.ASR_OCR,
                    JobStatus.FUSION,
                ]
            )
        )
        count = await self._execute_query(query, fetch="scalar")
        return count or 0

    async def can_start_new_job(self) -> bool:
        """Check if a new job can be started (under concurrency limit)."""
        active_count = await self.count_active_processing_jobs()
        return active_count < self.MAX_CONCURRENT_JOBS

    async def start_next_queued_job(self) -> Optional[str]:
        """
        Start the next queued job if there's capacity.
        Returns the job_id if a job was started, None otherwise.
        """
        if not await self.can_start_new_job():
            return None
        
        # Get the oldest queued job
        query = (
            sa.select(jobs_table)
            .where(jobs_table.c.status == JobStatus.QUEUED)
            .order_by(jobs_table.c.created_at.asc())
            .limit(1)
        )
        
        job_record = await self._execute_query(query, fetch="one")
        if not job_record:
            return None
        
        job_id = job_record["job_id"]
        scene_id = job_record["scene_id"]
        
        # Start processing
        logger.info("Starting queued job", job_id=job_id, scene_id=scene_id)
        try:
            # Start processing in background (don't await to avoid blocking)
            asyncio.create_task(
                self.start_processing_pipeline(job_id)
            )
            return job_id
        except Exception as e:
            logger.error("Failed to start queued job", job_id=job_id, error=str(e))
            await self._fail_job(job_id, f"Failed to start processing: {str(e)}")
            return None

    async def start_processing_pipeline(self, job_id: str):
        """Start the complete processing pipeline."""
        
        # Check if we can start processing (should already be checked, but double-check)
        if not await self.can_start_new_job():
            logger.warning("Cannot start job, at concurrency limit", job_id=job_id, max_concurrent=self.MAX_CONCURRENT_JOBS)
            # Keep job in QUEUED status - it will be picked up later
            return

        # CRITICAL: Update status to SAMPLING immediately after concurrency check
        await self._update_job_status(job_id, JobStatus.SAMPLING)

        try:
            # Get job record directly to access metadata and scene_id
            query = sa.select(jobs_table).where(jobs_table.c.job_id == job_id)
            job_record = await self._execute_query(query, fetch="one")
            if not job_record:
                await self._fail_job(job_id, "Job not found")
                return
            
            scene_id = job_record["scene_id"]
            metadata = job_record.get("metadata") or {}
            sample_fps = metadata.get("sample_fps")
            max_frames_per_scene = metadata.get("max_frames_per_scene")
            
            # Get scene data from our database
            from ..database import scenes_table
            scene_query = sa.select(scenes_table).where(scenes_table.c.scene_id == scene_id)
            scene_record = await self._execute_query(scene_query, fetch="one")
            
            scene = VideoScene(
                id=scene_id,
                title=scene_record.get("title") if scene_record else None,
                path=scene_record.get("path", "") if scene_record else "",
                duration=scene_record.get("duration_seconds") if scene_record else None,
                frame_rate=scene_record.get("frame_rate") if scene_record else None,
                width=None,
                height=None,
            )
            
            # Check for existing frames with matching config
            existing_frames = await self._check_existing_frames(scene.id, sample_fps, max_frames_per_scene)
            
            if existing_frames:
                # Reuse existing frames - skip sampling and embeddings
                old_job_id = existing_frames["old_job_id"]
                frame_ids = existing_frames["frame_ids"]
                frame_count = existing_frames["frame_count"]
                
                logger.info(
                    "Reusing existing frames, skipping sampling and embeddings",
                    job_id=job_id,
                    old_job_id=old_job_id,
                    frame_count=frame_count
                )
                
                # Remap frames to new job
                remapped = await self._remap_frames_to_job(old_job_id, job_id, scene.id)
                if not remapped:
                    logger.warning("Failed to remap frames, falling back to full pipeline", job_id=job_id)
                    # Fall through to normal pipeline
                else:
                    # Update progress to show reused frames
                    await self._update_job_progress(
                        job_id,
                        "sampling",
                        {
                            "status": "completed",
                            "frames_extracted": frame_count,
                            "current_step": f"Reused {frame_count} frames from previous job",
                            "reused": True
                        },
                    )
                    
                    await self._update_job_progress(
                        job_id,
                        "embeddings",
                        {
                            "status": "completed",
                            "embeddings_count": frame_count,
                            "frames_processed": frame_count,
                            "current_step": f"Reused {frame_count} embeddings from previous job",
                            "reused": True
                        },
                    )
                    
                    # Store reusable_frames in metadata so fusion worker can find them
                    await self._update_job_metadata(
                        job_id,
                        {
                            "reusable_frames": {
                                "frame_ids": frame_ids,
                                "original_job_id": old_job_id,
                                "frame_count": frame_count
                            }
                        }
                    )
                    
                    # Create dummy embeddings result for fusion
                    embeddings_result = {
                        "success": True,
                        "embedding_count": frame_count,
                        "frame_ids": frame_ids,
                        "model_name": getattr(get_settings(), 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'),
                        "reused": True
                    }
                    
                    # Create dummy ASR/OCR result for fusion (ASR/OCR disabled)
                    asr_ocr_result = {
                        "success": True,
                        "asr_segment_count": 0,
                        "ocr_text_count": 0,
                        "total_text_entries": 0
                    }
                    
                    # Go directly to fusion
                    await self._update_job_status(job_id, JobStatus.FUSION)
                    await self._update_job_progress(
                        job_id,
                        "fusion",
                        {
                            "status": "processing",
                            "current_step": "Analyzing frames and generating suggestions",
                            "tags_analyzed": 0,
                            "suggestions_generated": 0,
                        },
                    )
                    fusion_result = await self._queue_fusion_task(job_id, embeddings_result, asr_ocr_result)

                    if not fusion_result.get("success"):
                        error_msg = fusion_result.get("error", "Fusion step failed")
                        await self._fail_job(job_id, f"Fusion step failed: {error_msg}")
                        return

                    await self._update_job_progress(
                        job_id,
                        "fusion",
                        {
                            "status": "completed",
                            "suggestions_generated": fusion_result.get("suggestion_count", 0),
                            "high_confidence": fusion_result.get("high_confidence_count", 0),
                            "medium_confidence": fusion_result.get("medium_confidence_count", 0),
                            "low_confidence": fusion_result.get("low_confidence_count", 0),
                            "current_step": "Completed",
                        },
                    )
                    await self._update_job_status(job_id, JobStatus.COMPLETED)

                    logger.info(
                        "Job completed successfully (reused frames)",
                        job_id=job_id,
                        scene_id=scene.id,
                        suggestions_count=fusion_result.get("suggestion_count", 0),
                    )
                    
                    # Try to start next queued job now that this one is done
                    asyncio.create_task(self.start_next_queued_job())
                    return

            # No existing frames found, proceed with normal pipeline

            sampling_result = await self._queue_sampling_task(job_id, scene)

            if sampling_result.get("success"):
                await self._update_job_progress(
                    job_id,
                    "sampling",
                    {
                        "status": "completed",
                        "frames_extracted": sampling_result.get("frame_count", 0),
                        "current_step": "Completed",
                    },
                )

                await self._update_job_status(job_id, JobStatus.EMBEDDINGS)
                await self._update_job_progress(
                    job_id,
                    "embeddings",
                    {
                        "status": "processing",
                        "current_step": "Starting embeddings generation",
                        "frames_total": sampling_result.get("frame_count", 0),
                        "frames_processed": 0,
                    },
                )
                embeddings_task = self._queue_embeddings_task(job_id, sampling_result["frame_ids"])

                # ASR/OCR disabled - not useful for this use case and consumes too many resources
                # await self._update_job_status(job_id, JobStatus.ASR_OCR)
                # await self._update_job_progress(
                #     job_id,
                #     "asr_ocr",
                #     {
                #         "status": "processing",
                #         "current_step": "Extracting audio and text",
                #     },
                # )
                # asr_ocr_task = self._queue_asr_ocr_task(job_id, scene)

                # Only wait for embeddings (ASR/OCR disabled)
                embeddings_result = await embeddings_task

                # Check for exceptions
                if isinstance(embeddings_result, Exception):
                    await self._fail_job(job_id, f"Embeddings step failed: {str(embeddings_result)}")
                    return
                
                # Check success flags
                if not embeddings_result.get("success"):
                    error_msg = embeddings_result.get("error", "Embeddings step failed")
                    await self._fail_job(job_id, f"Embeddings step failed: {error_msg}")
                    return
                
                # Create dummy ASR/OCR result for fusion (ASR/OCR disabled)
                asr_ocr_result = {
                    "success": True,
                    "asr_segment_count": 0,
                    "ocr_text_count": 0,
                    "total_text_entries": 0
                }
                
                # Both succeeded, update progress
                await self._update_job_progress(
                    job_id,
                    "embeddings",
                    {
                        "status": "completed",
                        "embeddings_count": embeddings_result.get("embedding_count", 0),
                        "frames_processed": embeddings_result.get("embedding_count", 0),
                        "current_step": "Completed",
                    },
                )
                # ASR/OCR disabled - skip progress update
                # await self._update_job_progress(
                #     job_id,
                #     "asr_ocr",
                #     {
                #         "status": "completed",
                #         "text_segments": asr_ocr_result.get("total_text_entries", 0),
                #         "asr_segments": asr_ocr_result.get("asr_segment_count", 0),
                #         "ocr_texts": asr_ocr_result.get("ocr_text_count", 0),
                #         "current_step": "Completed",
                #     },
                # )

                await self._update_job_status(job_id, JobStatus.FUSION)
                await self._update_job_progress(
                    job_id,
                    "fusion",
                    {
                        "status": "processing",
                        "current_step": "Analyzing frames and generating suggestions",
                        "tags_analyzed": 0,
                        "suggestions_generated": 0,
                    },
                )
                fusion_result = await self._queue_fusion_task(job_id, embeddings_result, asr_ocr_result)

                if not fusion_result.get("success"):
                    error_msg = fusion_result.get("error", "Fusion step failed")
                    await self._fail_job(job_id, f"Fusion step failed: {error_msg}")
                    return

                await self._update_job_progress(
                    job_id,
                    "fusion",
                    {
                        "status": "completed",
                        "suggestions_generated": fusion_result.get("suggestion_count", 0),
                        "high_confidence": fusion_result.get("high_confidence_count", 0),
                        "medium_confidence": fusion_result.get("medium_confidence_count", 0),
                        "low_confidence": fusion_result.get("low_confidence_count", 0),
                        "current_step": "Completed",
                    },
                )
                await self._update_job_status(job_id, JobStatus.COMPLETED)

                logger.info(
                    "Job completed successfully",
                    job_id=job_id,
                    scene_id=scene.id,
                    suggestions_count=fusion_result.get("suggestion_count", 0),
                )
                
                # Try to start next queued job now that this one is done
                asyncio.create_task(self.start_next_queued_job())
            else:
                await self._fail_job(job_id, "Sampling step failed")
                # Try to start next queued job even on failure
                asyncio.create_task(self.start_next_queued_job())

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Job processing pipeline failed", job_id=job_id, error=str(exc))
            await self._fail_job(job_id, str(exc))
            # Try to start next queued job even on failure
            asyncio.create_task(self.start_next_queued_job())

    async def _queue_sampling_task(self, job_id: str, scene: VideoScene) -> Dict[str, Any]:
        """Queue frame sampling task."""

        logger.info("Queueing sampling task", job_id=job_id, scene_id=scene.id)
        task = celery_app.send_task(
            "app.sampler.extract_frames",
            args=[job_id, scene.dict()],
            priority=self._get_celery_priority(job_id),
        )
        
        logger.info("Sampling task queued, waiting for result", job_id=job_id, task_id=task.id)

        try:
            # Run the blocking task.get() in a thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                lambda: task.get(timeout=settings.job_timeout_seconds)
            )
            logger.info("Sampling task completed", job_id=job_id, result=result)
            return result
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Sampling task failed", job_id=job_id, error=str(exc), exc_type=type(exc).__name__)
            return {"success": False, "error": str(exc)}

    async def _queue_embeddings_task(self, job_id: str, frame_ids: List[str]) -> Dict[str, Any]:
        """Queue embeddings generation task."""

        task = celery_app.send_task(
            "app.embeddings.generate_embeddings",
            args=[job_id, frame_ids],
            priority=self._get_celery_priority(job_id),
        )

        try:
            return task.get(timeout=settings.job_timeout_seconds)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Embeddings task failed", job_id=job_id, error=str(exc))
            return {"success": False, "error": str(exc)}

    async def _queue_asr_ocr_task(self, job_id: str, scene: VideoScene) -> Dict[str, Any]:
        """Queue ASR/OCR processing task."""

        task = celery_app.send_task(
            "app.asr_ocr.process_audio_text",
            args=[job_id, scene.dict()],
            priority=self._get_celery_priority(job_id),
        )

        try:
            return task.get(timeout=settings.job_timeout_seconds)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("ASR/OCR task failed", job_id=job_id, error=str(exc))
            return {"success": False, "error": str(exc)}

    async def _queue_fusion_task(self, job_id: str, embeddings_result: Dict[str, Any], asr_ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Queue fusion and suggestion generation task."""

        logger.info("Queueing fusion task", job_id=job_id)
        task = celery_app.send_task(
            "app.fusion.generate_suggestions",
            args=[job_id, embeddings_result, asr_ocr_result],
            priority=self._get_celery_priority(job_id),
        )
        
        logger.info("Fusion task queued, waiting for result", job_id=job_id, task_id=task.id)

        try:
            # Run the blocking task.get() in a thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                lambda: task.get(timeout=settings.job_timeout_seconds)
            )
            logger.info("Fusion task completed", job_id=job_id, result=result)
            return result
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Fusion task failed", job_id=job_id, error=str(exc), exc_type=type(exc).__name__)
            return {"success": False, "error": str(exc)}

    def _get_celery_priority(self, job_id: str) -> int:  # pylint: disable=unused-argument
        """Get Celery priority based on job priority."""

        return 5  # Normal priority

    async def _get_recent_job(self, scene_id: str) -> Optional[Job]:
        """Get most recent completed job for a scene."""

        query = (
            sa.select(jobs_table)
            .where(
                jobs_table.c.scene_id == scene_id,
                jobs_table.c.status == JobStatus.COMPLETED,
            )
            .order_by(jobs_table.c.completed_at.desc())
            .limit(1)
        )

        record = await self._execute_query(query, fetch="one")
        return Job(**record) if record else None

    async def _check_existing_frames(
        self, 
        scene_id: str, 
        sample_fps: Optional[float], 
        max_frames_per_scene: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if frames and embeddings exist for a scene with matching configuration.
        
        Returns:
            Dict with 'old_job_id' and 'frame_ids' if found, None otherwise
        """
        try:
            # Normalize None values to defaults for comparison
            # This ensures backward compatibility with old jobs that don't have these fields
            default_sample_fps = settings.sample_fps
            default_max_frames = settings.max_frames_per_scene
            
            normalized_sample_fps = sample_fps if sample_fps is not None else default_sample_fps
            normalized_max_frames = max_frames_per_scene if max_frames_per_scene is not None else default_max_frames
            
            # Find completed jobs for this scene with matching metadata
            query = (
                sa.select(jobs_table)
                .where(
                    jobs_table.c.scene_id == scene_id,
                    jobs_table.c.status == JobStatus.COMPLETED,
                )
                .order_by(jobs_table.c.completed_at.desc())
            )
            
            completed_jobs = await self._execute_query(query, fetch="all") or []
            
            # Check each job for matching config
            for job_record in completed_jobs:
                metadata = job_record.get("metadata") or {}
                job_sample_fps = metadata.get("sample_fps")
                job_max_frames = metadata.get("max_frames_per_scene")
                
                # Normalize old job values to defaults if None (for backward compatibility)
                normalized_job_sample_fps = job_sample_fps if job_sample_fps is not None else default_sample_fps
                normalized_job_max_frames = job_max_frames if job_max_frames is not None else default_max_frames
                
                # Compare normalized values
                if normalized_job_sample_fps == normalized_sample_fps and normalized_job_max_frames == normalized_max_frames:
                    old_job_id = job_record["job_id"]
                    
                    # Check if this job has frames
                    frames_query = sa.select(frame_samples_table.c.id).where(
                        frame_samples_table.c.job_id == old_job_id
                    )
                    frame_records = await self._execute_query(frames_query, fetch="all") or []
                    frame_ids = [row["id"] for row in frame_records]
                    
                    if not frame_ids:
                        continue
                    
                    # Check if embeddings exist for these frames
                    # Get model name from settings
                    model_name = getattr(get_settings(), 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                    
                    embeddings_query = sa.select(embeddings_table.c.frame_id).where(
                        embeddings_table.c.frame_id.in_(frame_ids),
                        embeddings_table.c.model_name == model_name
                    ).distinct()
                    embedding_records = await self._execute_query(embeddings_query, fetch="all") or []
                    embeddings_frame_ids = {row["frame_id"] for row in embedding_records}
                    
                    # Check if all frames have embeddings
                    if len(embeddings_frame_ids) == len(frame_ids):
                        logger.info(
                            "Found existing frames and embeddings with matching config",
                            scene_id=scene_id,
                            old_job_id=old_job_id,
                            frame_count=len(frame_ids),
                            sample_fps=sample_fps,
                            max_frames_per_scene=max_frames_per_scene
                        )
                        return {
                            "old_job_id": old_job_id,
                            "frame_ids": frame_ids,
                            "frame_count": len(frame_ids)
                        }
            
            return None
            
        except Exception as e:
            logger.error("Failed to check for existing frames", scene_id=scene_id, error=str(e))
            return None

    async def _remap_frames_to_job(self, old_job_id: str, new_job_id: str, scene_id: str) -> bool:
        """
        Remap frames from old job to new job.
        
        Updates frame_samples.job_id from old_job_id to new_job_id.
        Embeddings don't need updating since they reference frame_id, not job_id.
        """
        try:
            async with async_session_maker() as session:
                # Update frame_samples to point to new job
                update_query = (
                    sa.update(frame_samples_table)
                    .where(
                        frame_samples_table.c.job_id == old_job_id,
                        frame_samples_table.c.scene_id == scene_id
                    )
                    .values(job_id=new_job_id)
                )
                result = await session.execute(update_query)
                await session.commit()
                
                remapped_count = result.rowcount
                logger.info(
                    "Remapped frames to new job",
                    old_job_id=old_job_id,
                    new_job_id=new_job_id,
                    scene_id=scene_id,
                    frame_count=remapped_count
                )
                return remapped_count > 0
                
        except Exception as e:
            logger.error(
                "Failed to remap frames",
                old_job_id=old_job_id,
                new_job_id=new_job_id,
                error=str(e)
            )
            return False

    async def _update_job_status(self, job_id: str, status: JobStatus):
        """Update job status."""

        update_data: Dict[str, Any] = {"status": status}

        if status in {JobStatus.SAMPLING, JobStatus.EMBEDDINGS, JobStatus.ASR_OCR, JobStatus.FUSION}:
            update_data["started_at"] = datetime.now(pytz.UTC)
        elif status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            update_data["completed_at"] = datetime.now(pytz.UTC)

        query = sa.update(jobs_table).where(jobs_table.c.job_id == job_id).values(**update_data)
        await self._execute_query(query, commit=True)

    async def _update_job_progress(self, job_id: str, step: str, progress_data: Dict[str, Any]):
        """Update job progress for a specific step."""

        query = sa.select(jobs_table).where(jobs_table.c.job_id == job_id)
        record = await self._execute_query(query, fetch="one")

        if not record:
            return

        current_progress = dict(record.get("progress") or {})
        current_progress[step] = progress_data

        update_query = (
            sa.update(jobs_table)
            .where(jobs_table.c.job_id == job_id)
            .values(progress=current_progress)
        )
        await self._execute_query(update_query, commit=True)
    
    async def _update_job_metadata(self, job_id: str, metadata_updates: Dict[str, Any]):
        """Update job metadata by merging with existing metadata."""
        
        query = sa.select(jobs_table).where(jobs_table.c.job_id == job_id)
        record = await self._execute_query(query, fetch="one")
        
        if not record:
            return
        
        current_metadata = dict(record.get("metadata") or {})
        current_metadata.update(metadata_updates)
        
        update_query = (
            sa.update(jobs_table)
            .where(jobs_table.c.job_id == job_id)
            .values(metadata=current_metadata)
        )
        await self._execute_query(update_query, commit=True)

    async def _fail_job(self, job_id: str, error_message: str):
        """Mark job as failed with error message."""

        query = (
            sa.update(jobs_table)
            .where(jobs_table.c.job_id == job_id)
            .values(
                status=JobStatus.FAILED,
                error_message=error_message,
                completed_at=datetime.now(pytz.UTC),
            )
        )

        await self._execute_query(query, commit=True)
        logger.error("Job failed", job_id=job_id, error=error_message)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's still active."""
        try:
            job = await self.get_job(job_id)
            if not job:
                return False
            
            # Only cancel if job is not already completed, failed, or cancelled
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                logger.warning("Cannot cancel job that is already finished", job_id=job_id, status=job.status.value)
                return False
            
            # Update job status to cancelled
            await self._update_job_status(job_id, JobStatus.CANCELLED)
            
            # Update error message to indicate cancellation
            query = (
                sa.update(jobs_table)
                .where(jobs_table.c.job_id == job_id)
                .values(error_message="Job cancelled by user")
            )
            await self._execute_query(query, commit=True)
            
            logger.info("Job cancelled", job_id=job_id)
            return True
            
        except Exception as e:
            logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            return False

    async def reset_excess_processing_jobs_to_queued(self) -> int:
        """
        Reset jobs in processing states that exceed the concurrency limit back to QUEUED.
        
        This is useful for jobs created before the concurrency limit was implemented.
        Only resets jobs that are beyond the MAX_CONCURRENT_JOBS limit.
        
        IMPORTANT: This only resets jobs that are truly excess. It does NOT reset jobs
        that are actually making progress (have started_at set and are recent).
        
        Returns:
            Number of jobs reset to QUEUED
        """
        try:
            # Get all jobs in processing states, ordered by creation time
            processing_states = [
                JobStatus.SAMPLING,
                JobStatus.EMBEDDINGS,
                JobStatus.ASR_OCR,
                JobStatus.FUSION,
            ]
            
            query = (
                sa.select(jobs_table)
                .where(jobs_table.c.status.in_(processing_states))
                .order_by(jobs_table.c.created_at.asc())
            )
            
            processing_jobs = await self._execute_query(query, fetch="all") or []
            
            if len(processing_jobs) <= self.MAX_CONCURRENT_JOBS:
                # No excess jobs
                return 0
            
            # Check which jobs are actually active (have started_at and are recent)
            # Jobs without started_at or with old started_at are likely stuck and can be reset
            active_jobs = []
            stuck_jobs = []
            
            for job_record in processing_jobs:
                started_at = job_record.get("started_at")
                # If job has started_at and it's recent (within last 30 minutes), consider it active
                if started_at:
                    time_since_start = datetime.now(pytz.UTC) - started_at
                    if time_since_start < timedelta(minutes=30):
                        active_jobs.append(job_record)
                    else:
                        stuck_jobs.append(job_record)
                else:
                    # No started_at means it never properly started, can be reset
                    stuck_jobs.append(job_record)
            
            # Keep active jobs, reset stuck jobs
            # But if we have more than MAX_CONCURRENT_JOBS active jobs, reset the excess
            if len(active_jobs) > self.MAX_CONCURRENT_JOBS:
                # Too many active jobs, reset the excess oldest ones
                excess_active = active_jobs[self.MAX_CONCURRENT_JOBS:]
                stuck_jobs.extend(excess_active)
                active_jobs = active_jobs[:self.MAX_CONCURRENT_JOBS]
            
            # Reset stuck/excess jobs to QUEUED
            reset_count = 0
            for job_record in stuck_jobs:
                job_id = job_record["job_id"]
                old_status = job_record["status"]
                
                try:
                    # Reset to QUEUED
                    await self._update_job_status(job_id, JobStatus.QUEUED)
                    reset_count += 1
                    logger.info(
                        "Reset excess/stuck processing job to queued",
                        job_id=job_id,
                        old_status=old_status.value,
                        max_concurrent=self.MAX_CONCURRENT_JOBS
                    )
                except Exception as e:
                    logger.error("Failed to reset job to queued", job_id=job_id, error=str(e))
            
            if reset_count > 0:
                logger.info(
                    "Reset excess processing jobs to queued",
                    reset_count=reset_count,
                    total_processing=len(processing_jobs),
                    active_jobs=len(active_jobs),
                    max_concurrent=self.MAX_CONCURRENT_JOBS
                )
            
            return reset_count
            
        except Exception as e:
            logger.error("Failed to reset excess processing jobs", error=str(e), exc_info=True)
            return 0

    async def start_queued_jobs_up_to_limit(self) -> int:
        """
        Start queued jobs up to the concurrency limit.
        This ensures that if there's capacity, queued jobs are started.
        
        Returns:
            Number of jobs started
        """
        started_count = 0
        max_to_start = self.MAX_CONCURRENT_JOBS - await self.count_active_processing_jobs()
        
        if max_to_start <= 0:
            return 0
        
        for _ in range(max_to_start):
            if not await self.can_start_new_job():
                break
            
            job_id = await self.start_next_queued_job()
            if job_id:
                started_count += 1
            else:
                # No more queued jobs
                break
        
        if started_count > 0:
            logger.info(
                "Started queued jobs up to limit",
                started_count=started_count,
                max_concurrent=self.MAX_CONCURRENT_JOBS
            )
        
        return started_count

    async def detect_and_reset_stuck_jobs(self, timeout_minutes: int = 60) -> int:
        """
        Detect and reset jobs that have been stuck in processing state for too long.
        
        This is useful after Docker restarts, when jobs may be in processing state
        but the workers have stopped.
        
        Args:
            timeout_minutes: Number of minutes a job can be in processing state before being considered stuck
            
        Returns:
            Number of stuck jobs that were reset
        """
        try:
            # Find jobs that are in processing state but haven't been updated recently
            processing_states = [
                JobStatus.SAMPLING,
                JobStatus.EMBEDDINGS,
                JobStatus.ASR_OCR,
                JobStatus.FUSION,
            ]
            
            timeout_threshold = datetime.now(pytz.UTC) - timedelta(minutes=timeout_minutes)
            
            # Find stuck jobs:
            # 1. Jobs in processing state with started_at older than timeout
            # 2. Jobs in processing state with no started_at (never properly started)
            # 3. Jobs in QUEUED state that are older than timeout (stuck in queue)
            query = (
                sa.select(jobs_table)
                .where(
                    sa.or_(
                        # Processing jobs that started too long ago
                        sa.and_(
                            jobs_table.c.status.in_(processing_states),
                            jobs_table.c.started_at.is_not(None),
                            jobs_table.c.started_at < timeout_threshold
                        ),
                        # Processing jobs that never started (started_at is None)
                        sa.and_(
                            jobs_table.c.status.in_(processing_states),
                            jobs_table.c.started_at.is_(None),
                            jobs_table.c.created_at < timeout_threshold
                        ),
                        # Queued jobs that are too old (likely stuck)
                        sa.and_(
                            jobs_table.c.status == JobStatus.QUEUED,
                            jobs_table.c.created_at < timeout_threshold
                        )
                    )
                )
            )
            
            stuck_jobs = await self._execute_query(query, fetch="all") or []
            
            if not stuck_jobs:
                logger.debug("No stuck jobs found")
                return 0
            
            logger.warning(
                "Found stuck jobs, resetting to failed state",
                stuck_count=len(stuck_jobs),
                timeout_minutes=timeout_minutes
            )
            
            # Reset each stuck job to failed state
            reset_count = 0
            for job_record in stuck_jobs:
                job_id = job_record["job_id"]
                status = job_record["status"]
                started_at = job_record.get("started_at")
                created_at = job_record.get("created_at")
                
                try:
                    # Determine how long the job has been stuck
                    if started_at:
                        stuck_duration = datetime.now(pytz.UTC) - started_at
                    elif created_at:
                        stuck_duration = datetime.now(pytz.UTC) - created_at
                    else:
                        stuck_duration = timedelta(minutes=0)
                    
                    error_msg = (
                        f"Job was stuck in {status} state for {stuck_duration.total_seconds() / 60:.1f} minutes "
                        f"(likely due to Docker restart or worker crash). Timeout: {timeout_minutes} minutes"
                    )
                    
                    await self._fail_job(job_id, error_msg)
                    reset_count += 1
                    logger.info(
                        "Reset stuck job",
                        job_id=job_id,
                        old_status=status,
                        stuck_duration_minutes=stuck_duration.total_seconds() / 60
                    )
                except Exception as e:
                    logger.error("Failed to reset stuck job", job_id=job_id, error=str(e))
            
            return reset_count
            
        except Exception as e:
            logger.error("Failed to detect and reset stuck jobs", error=str(e), exc_info=True)
            return 0

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and all related data."""
        try:
            job = await self.get_job(job_id)
            if not job:
                return False
            
            # Delete job and related data (cascade should handle related records)
            # But we'll delete explicitly to be safe
            from ..database import (
                suggestions_table,
                frame_samples_table,
                embeddings_table,
                text_analysis_table,
            )
            
            async with async_session_maker() as session:
                try:
                    # First get frame IDs before deleting frame_samples
                    frame_ids_query = sa.select(frame_samples_table.c.id).where(
                        frame_samples_table.c.job_id == job_id
                    )
                    frame_ids_result = await session.execute(frame_ids_query)
                    frame_ids_rows = frame_ids_result.fetchall()
                    frame_ids = [row[0] for row in frame_ids_rows] if frame_ids_rows else []
                    
                    # Delete in order (respecting foreign key constraints)
                    # 1. Delete suggestions (references job_id)
                    await session.execute(
                        sa.delete(suggestions_table).where(suggestions_table.c.job_id == job_id)
                    )
                    
                    # 2. Delete embeddings (references frame_id)
                    if frame_ids:
                        await session.execute(
                            sa.delete(embeddings_table).where(embeddings_table.c.frame_id.in_(frame_ids))
                        )
                    
                    # 3. Delete text analysis (references job_id)
                    await session.execute(
                        sa.delete(text_analysis_table).where(text_analysis_table.c.job_id == job_id)
                    )
                    
                    # 4. Delete frame samples (references job_id)
                    await session.execute(
                        sa.delete(frame_samples_table).where(frame_samples_table.c.job_id == job_id)
                    )
                    
                    # 5. Finally delete the job itself
                    await session.execute(
                        sa.delete(jobs_table).where(jobs_table.c.job_id == job_id)
                    )
                    
                    await session.commit()
                    logger.info("Job deleted", job_id=job_id)
                    return True
                except Exception as e:
                    await session.rollback()
                    logger.error("Failed to delete job data", job_id=job_id, error=str(e))
                    raise
            
        except Exception as e:
            logger.error("Failed to delete job", job_id=job_id, error=str(e))
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get job statistics."""

        status_counts: Dict[str, int] = {}
        for status in JobStatus:
            count_query = sa.select(sa.func.count()).select_from(jobs_table).where(jobs_table.c.status == status)
            count = await self._execute_query(count_query, fetch="scalar") or 0
            status_counts[status.value] = int(count)

        completed_query = (
            sa.select(jobs_table.c.started_at, jobs_table.c.completed_at)
            .where(
                jobs_table.c.status == JobStatus.COMPLETED,
                jobs_table.c.started_at.is_not(None),
                jobs_table.c.completed_at.is_not(None),
            )
        )
        completed_jobs = await self._execute_query(completed_query, fetch="all") or []

        processing_times = [
            (row["completed_at"] - row["started_at"]).total_seconds() / 60
            for row in completed_jobs
            if row.get("started_at") and row.get("completed_at")
        ]

        avg_processing_time = (
            sum(processing_times) / len(processing_times) if processing_times else None
        )

        return {
            "total_jobs": sum(status_counts.values()),
            "status_breakdown": status_counts,
            "average_processing_time_minutes": avg_processing_time,
        }