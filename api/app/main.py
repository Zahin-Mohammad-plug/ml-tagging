"""
ML Tagger API

A REST API service for orchestrating ML-based video tagging jobs.
Provides endpoints for ingesting videos, managing suggestions, and 
coordinating with background workers.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import structlog
from typing import List, Optional, Any, Dict
from pydantic import BaseModel
import os
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from pathlib import Path
import json
import tempfile
import numpy as np

from .config import get_settings, Settings
from .database import init_db, get_database, frame_samples_table, async_session_maker, jobs_table, suggestions_table
from .models import (
    HealthResponse, 
    IngestRequest, 
    IngestResponse,
    Suggestion,
    SuggestionStatus,
    SuggestionResponse,
    ApprovalRequest,
    JobStatus,
    TextBasedSuggestionsRequest,
    TextBasedSuggestionsResponse,
    TextBasedSuggestion
)
from .services.job_orchestrator import JobOrchestrator
from .services.suggestion_service import SuggestionService
from .services.tag_sync import sync_tags_from_prompts, export_tags_to_prompts
from .services.text_tag_suggester import TextTagSuggester
from .services.settings_service import SettingsService, get_settings_service
from .services.tags_service import TagsService, get_tags_service
from .database import tags_table, blacklisted_tags_table
from .auth import get_current_user
from .exceptions import handle_exceptions

logger = structlog.get_logger()

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting ML Tagger API")
    await init_db()
    
    # Sync tags from tag_prompts.json on startup
    try:
        sync_result = await sync_tags_from_prompts()
        if sync_result.get("success"):
            logger.info(
                "Tag sync completed on startup",
                tags_created=sync_result.get("tags_created", 0),
                tags_updated=sync_result.get("tags_updated", 0),
                total_tags=sync_result.get("total_in_prompts", 0)
            )
        else:
            logger.warning("Tag sync failed on startup", error=sync_result.get("error"))
    except Exception as e:
        logger.error("Tag sync error on startup", error=str(e), exc_info=True)
    
    # Detect and reset stuck jobs (e.g., after Docker restart)
    try:
        orchestrator = JobOrchestrator()
        stuck_count = await orchestrator.detect_and_reset_stuck_jobs(timeout_minutes=60)
        if stuck_count > 0:
            logger.warning(
                "Reset stuck jobs on startup",
                stuck_count=stuck_count,
                message="Jobs were likely interrupted by a Docker restart or worker crash"
            )
        else:
            logger.debug("No stuck jobs found on startup")
    except Exception as e:
        logger.error("Failed to detect stuck jobs on startup", error=str(e), exc_info=True)
    
    # Reset excess processing jobs to QUEUED (for jobs created before concurrency limit)
    try:
        orchestrator = JobOrchestrator()
        reset_count = await orchestrator.reset_excess_processing_jobs_to_queued()
        if reset_count > 0:
            logger.info(
                "Reset excess processing jobs to queued on startup",
                reset_count=reset_count,
                max_concurrent=orchestrator.MAX_CONCURRENT_JOBS,
                message="Jobs created before concurrency limit was implemented"
            )
    except Exception as e:
        logger.error("Failed to reset excess processing jobs on startup", error=str(e), exc_info=True)
    
    # Start queued jobs up to the concurrency limit
    try:
        orchestrator = JobOrchestrator()
        started_count = await orchestrator.start_queued_jobs_up_to_limit()
        if started_count > 0:
            logger.info(
                "Started queued jobs on startup",
                started_count=started_count,
                max_concurrent=orchestrator.MAX_CONCURRENT_JOBS
            )
    except Exception as e:
        logger.error("Failed to start queued jobs on startup", error=str(e), exc_info=True)
    
    yield
    # Shutdown
    logger.info("Shutting down ML Tagger API")
    # SQLAlchemy async engine cleanup happens automatically

# Create FastAPI app
app = FastAPI(
    title="ML Tagger API",
    description="ML-powered video tagging with human-in-the-loop review",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
handle_exceptions(app)

# Dependency injection
def get_job_orchestrator():
    return JobOrchestrator()

def get_suggestion_service():
    return SuggestionService()

def get_text_tag_suggester():
    return TextTagSuggester()

@app.get("/health/clip")
async def health_clip():
    """
    Health check endpoint for CLIP model verification.
    Tests that CLIP embedder can load and encode text/images.
    """
    try:
        # Try to import and load CLIP embedder
        from workers.app.embeddings import CLIPEmbedder
        from workers.app.config import get_worker_settings
        
        settings = get_worker_settings()
        model_name = getattr(settings, 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        
        try:
            embedder = CLIPEmbedder(model_name=model_name)
            
            # Test text encoding
            test_text = "test prompt for health check"
            test_embedding = embedder.encode_text(test_text)
            embedding_dim = len(test_embedding)
            device = embedder.device
            
            # Compute a simple similarity test
            test_text2 = "similar test prompt"
            test_embedding2 = embedder.encode_text(test_text2)
            similarity = float(np.dot(
                test_embedding / (np.linalg.norm(test_embedding) + 1e-8),
                test_embedding2 / (np.linalg.norm(test_embedding2) + 1e-8)
            ))
            
            return {
                "status": "healthy",
                "clip_model": {
                    "model_name": model_name,
                    "device": device,
                    "embedding_dim": embedding_dim,
                    "test_encoding_successful": True,
                    "test_similarity": round(similarity, 4),
                    "message": "CLIP embedder is loaded and working correctly"
                }
            }
        except Exception as load_error:
            return {
                "status": "unhealthy",
                "clip_model": {
                    "model_name": model_name,
                    "error": str(load_error),
                    "message": "Failed to load or test CLIP embedder"
                }
            }
    except ImportError as import_error:
        return {
            "status": "error",
            "error": f"Failed to import CLIP embedder: {str(import_error)}",
            "message": "CLIP embedder module not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Unexpected error during CLIP health check"
        }

@app.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_database)):
    """Health check endpoint"""
    try:
        # Check database connection
        result = await db.execute(sa.text("SELECT 1"))
        result.scalar()
        db_status = "healthy"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        db_status = "unhealthy"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" else "unhealthy",
        database=db_status,
        queue="unknown",  # Will implement Redis check
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest_scene(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Ingest a video for ML processing
    
    This endpoint:
    1. Validates the video exists in the database or creates a record
    2. Creates a processing job
    3. Queues background workers for sampling, embeddings, ASR/OCR, and fusion
    """
    logger.info("Ingesting video for processing", scene_id=request.scene_id)
    
    try:
        # Create scene record in database if it doesn't exist
        from .database import scenes_table, async_session_maker
        async with async_session_maker() as session:
            try:
                # Check if scene exists
                check_query = sa.select(scenes_table).where(scenes_table.c.scene_id == request.scene_id)
                result = await session.execute(check_query)
                existing_scene = result.first()
                
                if not existing_scene:
                    # Insert scene record from request data
                    insert_query = scenes_table.insert().values(
                        scene_id=request.scene_id,
                        title=request.title or f"Video {request.scene_id}",
                        path=request.path,
                        duration_seconds=request.duration,
                        frame_rate=request.frame_rate
                    )
                    await session.execute(insert_query)
                    await session.commit()
                    logger.info("Created scene record", scene_id=request.scene_id)
            except Exception as e:
                await session.rollback()
                logger.warning("Failed to create scene record", error=str(e))
                # Continue anyway - scene might exist from concurrent request
        
        # Handle clean_process: delete all old jobs for this scene
        if request.clean_process:
            from .database import jobs_table
            async with async_session_maker() as session:
                try:
                    delete_query = sa.delete(jobs_table).where(
                        jobs_table.c.scene_id == request.scene_id
                    )
                    result = await session.execute(delete_query)
                    await session.commit()
                    deleted_count = result.rowcount
                    logger.info(
                        "Deleted old jobs for clean process",
                        scene_id=request.scene_id,
                        deleted_count=deleted_count
                    )
                except Exception as e:
                    await session.rollback()
                    logger.warning("Failed to delete old jobs", scene_id=request.scene_id, error=str(e))
                    # Continue anyway - might be no jobs to delete
        
        # Check if already processing or recently processed (after clean_process)
        existing_job = await orchestrator.get_active_job(request.scene_id)
        if existing_job:
            return IngestResponse(
                job_id=existing_job.job_id,
                status="already_processing",
                message=f"Scene {request.scene_id} is already being processed"
            )
        
        # Create and queue processing job
        job = await orchestrator.create_job(
            scene_id=request.scene_id,
            priority=request.priority,
            force_reprocess=request.force_reprocess,
            max_frames_per_scene=request.max_frames_per_scene,
            sample_fps=request.sample_fps,
            auto_approve_threshold=request.auto_approve_threshold,
            auto_delete_threshold=request.auto_delete_threshold,
        )
        
        # Check if we can start processing immediately
        can_start = await orchestrator.can_start_new_job()
        if can_start:
            # Start processing pipeline immediately
            background_tasks.add_task(
                orchestrator.start_processing_pipeline,
                job.job_id,
            )
        else:
            # Job will remain in QUEUED status and be picked up when a slot opens
            logger.info(
                "Job queued (at concurrency limit)",
                job_id=job.job_id,
                scene_id=request.scene_id,
                max_concurrent=orchestrator.MAX_CONCURRENT_JOBS
            )
            # Try to start queued jobs in background (in case other jobs just completed)
            background_tasks.add_task(
                orchestrator.start_next_queued_job,
            )
        
        logger.info("Video processing job created", job_id=job.job_id, scene_id=request.scene_id)
        
        return IngestResponse(
            job_id=job.job_id,
            status="queued",
            message=f"Processing started for video {request.scene_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to ingest video", scene_id=request.scene_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/jobs/active")
async def get_active_jobs(
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Get all active jobs (queued, sampling, embeddings, fusion) in a single call.
    This is more efficient than fetching each job individually.
    NOTE: This route must come BEFORE /jobs/{job_id} to avoid route conflicts.
    """
    try:
        # Get all jobs and filter for active ones
        all_jobs = await orchestrator.list_jobs(limit=200)  # Get more jobs to include all active ones
        
        # Filter for active statuses
        active_statuses = [
            JobStatus.QUEUED,
            JobStatus.SAMPLING,
            JobStatus.EMBEDDINGS,
            JobStatus.ASR_OCR,
            JobStatus.FUSION,
        ]
        
        active_jobs = [job for job in all_jobs if job.status in active_statuses]
        
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "scene_id": job.scene_id,
                    "status": job.status.value,
                    "priority": job.priority.value if job.priority else "normal",
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message,
                    "progress": job.progress,
                }
                for job in active_jobs
            ],
            "count": len(active_jobs)
        }
    except Exception as e:
        logger.error("Failed to get active jobs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/jobs/active")
async def cancel_all_active_jobs(
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Cancel ALL active jobs (queued, sampling, embeddings, fusion).
    
    This will cancel all jobs that are currently queued or processing.
    Use with caution!
    """
    try:
        # Get all active jobs
        all_jobs = await orchestrator.list_jobs(limit=200)
        
        # Filter for active statuses
        active_statuses = [
            JobStatus.QUEUED,
            JobStatus.SAMPLING,
            JobStatus.EMBEDDINGS,
            JobStatus.ASR_OCR,
            JobStatus.FUSION,
        ]
        
        active_jobs = [job for job in all_jobs if job.status in active_statuses]
        
        if not active_jobs:
            return {
                "success": True,
                "cancelled_count": 0,
                "message": "No active jobs to cancel"
            }
        
        # Cancel each active job
        cancelled_count = 0
        failed_count = 0
        for job in active_jobs:
            success = await orchestrator.cancel_job(job.job_id)
            if success:
                cancelled_count += 1
            else:
                failed_count += 1
        
        logger.info("Cancelled all active jobs", cancelled=cancelled_count, failed=failed_count)
        
        return {
            "success": True,
            "cancelled_count": cancelled_count,
            "failed_count": failed_count,
            "total_active": len(active_jobs),
            "message": f"Cancelled {cancelled_count} active job(s)" + (f" ({failed_count} failed)" if failed_count > 0 else "")
        }
        
    except Exception as e:
        logger.error("Failed to cancel all active jobs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator)
):
    """Get real-time job processing status"""
    try:
        job = await orchestrator.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job.job_id,
            "scene_id": job.scene_id,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator)
):
    """Cancel a processing job"""
    try:
        success = await orchestrator.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled (may already be completed, failed, or cancelled)")
        
        return {
            "status": "cancelled",
            "message": f"Job {job_id} has been cancelled",
            "job_id": job_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator)
):
    """Delete a job and all related data"""
    try:
        success = await orchestrator.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "status": "deleted",
            "message": f"Job {job_id} has been deleted",
            "job_id": job_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator)
):
    """List recent processing jobs"""
    try:
        jobs = await orchestrator.list_jobs(status=status, limit=limit)
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "scene_id": job.scene_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                }
                for job in jobs
            ]
        }
    except Exception as e:
        logger.error("Failed to list jobs", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/suggestions", response_model=List[SuggestionResponse])
async def get_suggestions(
    status: Optional[SuggestionStatus] = None,
    scene_id: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Get suggestions with optional filtering, sorting, and pagination
    
    Returns suggestions with evidence frames and confidence details.
    
    Args:
        status: Filter by suggestion status (pending, approved, rejected)
        scene_id: Filter by scene ID
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        limit: Maximum number of suggestions to return
        offset: Number of suggestions to skip
        sort_by: Sort field (confidence, date, scene)
        sort_order: Sort order (asc, desc) - defaults to desc
    """
    try:
        suggestions = await suggestion_service.get_suggestions(
            status=status,
            scene_id=scene_id,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return suggestions
        
    except Exception as e:
        import traceback
        logger.error("Failed to fetch suggestions", error=str(e), error_type=type(e).__name__, traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/suggestions/text-based/{scene_id}", response_model=TextBasedSuggestionsResponse)
async def get_text_based_suggestions(
    scene_id: str,
    request: TextBasedSuggestionsRequest = Body(...),
    text_suggester: TextTagSuggester = Depends(get_text_tag_suggester),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Generate tag suggestions based on scene description, title, and optionally OCR text.
    
    Uses CLIP text encoder to match scene text with tag embeddings.
    Excludes tags that already exist on the scene.
    
    Args:
        scene_id: Video scene ID
        request: Configuration for text-based suggestions
    
    Returns:
        TextBasedSuggestionsResponse with tag suggestions and metadata
    """
    try:
        # Get scene from database
        from .database import scenes_table, async_session_maker
        async with async_session_maker() as session:
            scene_query = sa.select(scenes_table).where(scenes_table.c.scene_id == scene_id)
            result = await session.execute(scene_query)
            scene_row = result.first()
        
        if not scene_row:
            raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
        
        # Build a simple scene object for the text suggester
        class SceneData:
            def __init__(self, row):
                self.id = row.scene_id
                self.title = row.title
                self.description = getattr(row, 'description', None)
                self.path = row.path
                self.tags = []
        
        scene = SceneData(scene_row)
        
        # Generate suggestions
        suggestions = await text_suggester.suggest_tags(
            scene=scene,
            existing_tag_ids=None,
            use_description=request.use_description,
            use_title=request.use_title,
            use_ocr=request.use_ocr,
            min_confidence=request.min_confidence,
            max_suggestions=request.max_suggestions,
        )
        
        # Convert to response format
        suggestion_responses = [
            TextBasedSuggestion(
                tag_id=s["tag_id"],
                tag_name=s["tag_name"],
                confidence=s["confidence"],
                source=s.get("source", "text"),
                text_type=s.get("text_type")
            )
            for s in suggestions
        ]
        
        # Get total tags checked (approximate - we checked all active tags with embeddings)
        from .database import async_session_maker, tags_table
        async with async_session_maker() as session:
            count_query = sa.select(sa.func.count()).select_from(tags_table).where(
                tags_table.c.is_active == True,
                tags_table.c.embedding.is_not(None)
            )
            result = await session.execute(count_query)
            total_tags_checked = result.scalar() or 0
        
        return TextBasedSuggestionsResponse(
            scene_id=scene_id,
            suggestions=suggestion_responses,
            text_used={
                "description": request.use_description and bool(scene.description),
                "title": request.use_title and bool(scene.title),
                "ocr": request.use_ocr
            },
            total_tags_checked=total_tags_checked
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate text-based suggestions", scene_id=scene_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/suggestions/scenes")
async def get_scenes_with_suggestions(
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Get list of unique scenes that have suggestions, with scene titles
    Useful for populating scene filter dropdowns
    """
    try:
        # Get unique scene IDs from suggestions
        from .database import async_session_maker, suggestions_table, scenes_table
        import sqlalchemy as sa
        
        async with async_session_maker() as session:
            # Get distinct scene IDs from suggestions
            scene_ids_query = sa.select(
                suggestions_table.c.scene_id.distinct()
            )
            result = await session.execute(scene_ids_query)
            scene_ids = [row[0] for row in result.fetchall()]
            
            logger.info("Found scenes with suggestions", scene_count=len(scene_ids), scene_ids=scene_ids[:10])  # Log first 10
            
            # Get scene titles from scenes table
            scenes_with_titles = []
            for scene_id in scene_ids:
                try:
                    # Try to get title from scenes table first
                    scene_query = sa.select(scenes_table.c.title).where(
                        scenes_table.c.scene_id == scene_id
                    )
                    scene_result = await session.execute(scene_query)
                    scene_row = scene_result.first()
                    
                    title = scene_row[0] if scene_row and scene_row[0] else None
                    
                    scenes_with_titles.append({
                        "scene_id": scene_id,
                        "title": title or f"Scene {scene_id}"
                    })
                except Exception as e:
                    logger.warning("Failed to get scene title", scene_id=scene_id, error=str(e))
                    scenes_with_titles.append({
                        "scene_id": scene_id,
                        "title": f"Scene {scene_id}"
                    })
            
            # Sort by scene_id (numeric if possible)
            scenes_with_titles.sort(key=lambda x: (
                int(x["scene_id"]) if x["scene_id"].isdigit() else float('inf'),
                x["scene_id"]
            ))
            
            logger.info("Returning scenes with suggestions", count=len(scenes_with_titles))
            return scenes_with_titles
            
    except Exception as e:
        logger.error("Failed to fetch scenes with suggestions", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/suggestions/{suggestion_id}", response_model=SuggestionResponse)
async def get_suggestion(
    suggestion_id: str,
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get a specific suggestion by ID"""
    try:
        suggestion = await suggestion_service.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        return suggestion
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch suggestion", suggestion_id=suggestion_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/suggestions/{suggestion_id}/approve")
async def approve_suggestion(
    suggestion_id: str,
    request: ApprovalRequest,
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Approve a suggestion
    
    This will:
    1. Mark the suggestion as approved in the database
    2. Log the approval for audit purposes
    """
    try:
        suggestion = await suggestion_service.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        if suggestion.status != SuggestionStatus.PENDING:
            raise HTTPException(status_code=400, detail="Suggestion is not pending")
        
        # Only apply primary (non-backup) tags
        # Backup tags are stored for search/filtering but not applied as primary tags
        if suggestion.is_backup:
            raise HTTPException(
                status_code=400, 
                detail="Cannot approve backup tags. Backup tags are automatically excluded from primary tag application."
            )
        
        # Get tag name from tag_context
        tag_name = suggestion.tag_context.tag_name
        
        # Mark as approved
        await suggestion_service.approve_suggestion(
            suggestion_id=suggestion_id,
            approved_by=request.approved_by or "api_user",
            notes=request.notes
        )
        
        logger.info(
            "Suggestion approved and applied",
            suggestion_id=suggestion_id,
            scene_id=suggestion.scene_id,
            tag_name=tag_name
        )
        
        return {"status": "approved", "message": "Suggestion approved"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to approve suggestion", suggestion_id=suggestion_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/suggestions/{suggestion_id}/reject")
async def reject_suggestion(
    suggestion_id: str,
    request: ApprovalRequest,
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Reject a suggestion
    
    This will mark the suggestion as rejected for audit purposes.
    """
    try:
        suggestion = await suggestion_service.get_suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        
        if suggestion.status != SuggestionStatus.PENDING:
            raise HTTPException(status_code=400, detail="Suggestion is not pending")
        
        # Mark as rejected
        await suggestion_service.reject_suggestion(
            suggestion_id=suggestion_id,
            rejected_by=request.approved_by or "api_user",  # Reuse field
            notes=request.notes
        )
        
        # Get tag name from tag_context
        tag_name = suggestion.tag_context.tag_name
        
        logger.info(
            "Suggestion rejected",
            suggestion_id=suggestion_id,
            scene_id=suggestion.scene_id,
            tag_name=tag_name
        )
        
        return {"status": "rejected", "message": "Suggestion rejected"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reject suggestion", suggestion_id=suggestion_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/suggestions/bulk")
async def bulk_delete_suggestions(
    suggestion_ids: List[str] = Body(..., description="List of suggestion IDs to delete"),
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Delete multiple suggestions by IDs
    
    This will permanently delete the suggestions from the database.
    """
    try:
        if not suggestion_ids:
            raise HTTPException(status_code=400, detail="No suggestion IDs provided")
        
        result = await suggestion_service.delete_suggestions(suggestion_ids)
        
        logger.info(
            "Bulk delete completed",
            deleted_count=result["deleted_count"],
            total_requested=len(suggestion_ids)
        )
        
        return {
            "success": True,
            "deleted_count": result["deleted_count"],
            "message": f"Deleted {result['deleted_count']} suggestion(s)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete suggestions", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/suggestions/all")
async def delete_all_suggestions(
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Delete ALL suggestions from the database.
    
    This will permanently delete all suggestions regardless of status.
    Use with caution!
    """
    try:
        from .database import async_session_maker, suggestions_table
        import sqlalchemy as sa
        
        async with async_session_maker() as session:
            # Count suggestions before deletion
            count_query = sa.select(sa.func.count()).select_from(suggestions_table)
            count_result = await session.execute(count_query)
            total_count = count_result.scalar() or 0
            
            if total_count == 0:
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": "No suggestions to delete"
                }
            
            # Delete all suggestions
            delete_query = sa.delete(suggestions_table)
            result = await session.execute(delete_query)
            await session.commit()
            
            deleted_count = result.rowcount
            
            logger.info("Deleted all suggestions", deleted_count=deleted_count)
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Deleted all {deleted_count} suggestion(s)"
            }
        
    except Exception as e:
        logger.error("Failed to delete all suggestions", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/scenes/{scene_id}/suggestions")
async def delete_suggestions_for_scene(
    scene_id: str,
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Delete all suggestions and jobs for a specific scene
    
    This will:
    1. Delete all jobs for the scene (cascades to frames, embeddings, text_analysis)
    2. Delete all suggestions for the scene (cascades to audit_log)
    
    Useful for reprocessing a scene with fresh analysis.
    """
    try:
        async with async_session_maker() as session:
            # Count jobs and suggestions before deletion
            jobs_count = await session.execute(
                sa.select(sa.func.count())
                .select_from(jobs_table)
                .where(jobs_table.c.scene_id == scene_id)
            )
            jobs_total = jobs_count.scalar() or 0
            
            suggestions_count = await session.execute(
                sa.select(sa.func.count())
                .select_from(suggestions_table)
                .where(suggestions_table.c.scene_id == scene_id)
            )
            suggestions_total = suggestions_count.scalar() or 0
            
            if jobs_total == 0:
                return {
                    "status": "success",
                    "message": f"No jobs or suggestions found for scene {scene_id}",
                    "jobs_deleted": 0,
                    "suggestions_deleted": 0
                }
            
            # Delete jobs for this scene (cascade will handle related data)
            await session.execute(
                sa.delete(jobs_table).where(jobs_table.c.scene_id == scene_id)
            )
            
            await session.commit()
            
            logger.info(
                "Deleted suggestions and jobs for scene",
                scene_id=scene_id,
                jobs_deleted=jobs_total,
                suggestions_deleted=suggestions_total
            )
            
            return {
                "status": "success",
                "message": f"Deleted {jobs_total} jobs and {suggestions_total} suggestions for scene {scene_id}",
                "jobs_deleted": jobs_total,
                "suggestions_deleted": suggestions_total
            }
        
    except Exception as e:
        logger.error("Failed to delete suggestions for scene", scene_id=scene_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/scenes/{scene_id}/reprocess")
async def reprocess_scene(
    scene_id: str,
    background_tasks: BackgroundTasks,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Reprocess a scene by deleting old suggestions and creating a new job
    
    This will:
    1. Delete all existing jobs and suggestions for the scene
    2. Create a new ingest job for the scene
    3. Return the new job ID
    """
    try:
        # First, delete existing suggestions and jobs
        async with async_session_maker() as session:
            # Delete jobs for this scene (cascade will handle related data)
            await session.execute(
                sa.delete(jobs_table).where(jobs_table.c.scene_id == scene_id)
            )
            await session.commit()
        
        logger.info("Deleted old jobs and suggestions for scene", scene_id=scene_id)
        
        # Create new job with force_reprocess flag
        job = await orchestrator.create_job(
            scene_id=scene_id,
            priority="normal",
            force_reprocess=True
        )
        
        # Check if we can start processing immediately
        can_start = await orchestrator.can_start_new_job()
        if can_start:
            # Start processing pipeline immediately
            background_tasks.add_task(
                orchestrator.start_processing_pipeline,
                job.job_id,
            )
        else:
            # Job will remain in QUEUED status and be picked up when a slot opens
            logger.info(
                "Reprocess job queued (at concurrency limit)",
                job_id=job.job_id,
                scene_id=scene_id,
                max_concurrent=orchestrator.MAX_CONCURRENT_JOBS
            )
            # Try to start queued jobs in background
            background_tasks.add_task(
                orchestrator.start_next_queued_job,
            )
        
        logger.info(
            "Reprocessing scene",
            scene_id=scene_id,
            job_id=job.job_id
        )
        
        return {
            "status": "success",
            "message": f"Scene {scene_id} is being reprocessed",
            "job_id": job.job_id,
            "scene_id": scene_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reprocess scene", scene_id=scene_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reprocess scene: {str(e)}")

@app.post("/jobs/reset-stuck")
async def reset_stuck_jobs(
    timeout_minutes: int = 60,
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Manually detect and reset stuck jobs.
    
    Useful for recovering from Docker restarts or worker crashes.
    
    Args:
        timeout_minutes: Number of minutes a job can be in processing state before being considered stuck
    """
    try:
        reset_count = await orchestrator.detect_and_reset_stuck_jobs(timeout_minutes=timeout_minutes)
        return {
            "status": "success",
            "message": f"Reset {reset_count} stuck job(s)",
            "reset_count": reset_count
        }
    except Exception as e:
        logger.error("Failed to reset stuck jobs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset stuck jobs: {str(e)}")

@app.post("/jobs/reset-excess-to-queued")
async def reset_excess_to_queued(
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Reset jobs in processing states that exceed the concurrency limit back to QUEUED.
    
    Useful for jobs created before the concurrency limit was implemented.
    Only resets jobs beyond the MAX_CONCURRENT_JOBS limit.
    """
    try:
        reset_count = await orchestrator.reset_excess_processing_jobs_to_queued()
        return {
            "status": "success",
            "message": f"Reset {reset_count} excess processing job(s) to queued",
            "reset_count": reset_count,
            "max_concurrent": orchestrator.MAX_CONCURRENT_JOBS
        }
    except Exception as e:
        logger.error("Failed to reset excess jobs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset excess jobs: {str(e)}")

@app.post("/jobs/start-queued")
async def start_queued_jobs(
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """
    Manually start queued jobs up to the concurrency limit.
    
    Useful if jobs are stuck in queued state and not starting automatically.
    """
    try:
        started_count = await orchestrator.start_queued_jobs_up_to_limit()
        active_count = await orchestrator.count_active_processing_jobs()
        return {
            "status": "success",
            "message": f"Started {started_count} queued job(s)",
            "started_count": started_count,
            "active_jobs": active_count,
            "max_concurrent": orchestrator.MAX_CONCURRENT_JOBS
        }
    except Exception as e:
        logger.error("Failed to start queued jobs", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start queued jobs: {str(e)}")

@app.get("/stats")
async def get_stats(
    suggestion_service: SuggestionService = Depends(get_suggestion_service),
    orchestrator: JobOrchestrator = Depends(get_job_orchestrator),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get system statistics"""
    try:
        stats = await suggestion_service.get_stats()
        job_stats = await orchestrator.get_stats()
        
        return {
            "suggestions": stats,
            "jobs": job_stats,
            "system": {
                "version": "1.0.0",
                "uptime": "TODO"  # Calculate uptime
            }
        }
        
    except Exception as e:
        logger.error("Failed to fetch stats", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/settings")
async def get_settings_endpoint(
    settings_service: SettingsService = Depends(get_settings_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get all settings"""
    try:
        # user_id = current_user.id if current_user else None
        user_id = None  # For now, no user support
        settings = await settings_service.get_all_settings(user_id=user_id)
        return settings
    except Exception as e:
        logger.error("Failed to fetch settings", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/settings/{key}")
async def get_setting(
    key: str,
    settings_service: SettingsService = Depends(get_settings_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get a specific setting by key"""
    try:
        # user_id = current_user.id if current_user else None
        user_id = None  # For now, no user support
        value = await settings_service.get_setting(key, user_id=user_id)
        return {"key": key, "value": value}
    except Exception as e:
        logger.error("Failed to fetch setting", key=key, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/settings/{key}")
async def update_setting(
    key: str,
    value: Any = Body(...),
    settings_service: SettingsService = Depends(get_settings_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Update a specific setting"""
    try:
        # user_id = current_user.id if current_user else None
        user_id = None  # For now, no user support
        result = await settings_service.set_setting(key, value, user_id=user_id)
        return result
    except Exception as e:
        logger.error("Failed to update setting", key=key, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/settings")
async def update_settings(
    settings: dict,
    settings_service: SettingsService = Depends(get_settings_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Bulk update settings"""
    try:
        # user_id = current_user.id if current_user else None
        user_id = None  # For now, no user support
        result = await settings_service.set_settings(settings, user_id=user_id)
        return result
    except Exception as e:
        logger.error("Failed to update settings", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Tags API endpoints
@app.get("/tags")
async def get_tags(
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "name",
    sort_order: str = "asc",
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get tags with search and filter support"""
    try:
        tags = await tags_service.get_tags(
            search=search,
            is_active=is_active,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return tags
    except Exception as e:
        logger.error("Failed to fetch tags", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tags/export")
async def export_tags(
    include_prompts: bool = False,
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Export tags to JSON format"""
    try:
        tags = await tags_service.get_tags(limit=10000, is_active=None)
        
        if include_prompts:
            # Export in tag_prompts.json format
            tag_prompts = {}
            for tag in tags:
                if tag.get('name') and tag.get('prompts'):
                    tag_prompts[tag['name']] = tag['prompts']
            return {
                "success": True,
                "format": "tag_prompts",
                "tags": tag_prompts,
                "count": len(tag_prompts)
            }
        else:
            # Export just tag metadata (without prompts)
            tags_export = []
            for tag in tags:
                tags_export.append({
                    "tag_id": tag.get('tag_id'),
                    "name": tag.get('name'),
                    "description": tag.get('description'),
                    "is_active": tag.get('is_active'),
                    "is_blacklisted": tag.get('is_blacklisted'),
                    "review_threshold": tag.get('review_threshold'),
                    "auto_threshold": tag.get('auto_threshold'),
                    "aliases": tag.get('aliases', tag.get('synonyms', [])),
                    "parent_tag_ids": tag.get('parent_tag_ids', []),
                    "child_tag_ids": tag.get('child_tag_ids', []),
                })
            return {
                "success": True,
                "format": "tags_metadata",
                "tags": tags_export,
                "count": len(tags_export)
            }
    except Exception as e:
        logger.error("Failed to export tags", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tags/{tag_id}")
async def get_tag(
    tag_id: str,
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get a specific tag by ID"""
    try:
        tag = await tags_service.get_tag(tag_id)
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        return tag
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch tag", tag_id=tag_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/tags/{tag_id}")
async def update_tag(
    tag_id: str,
    tag_data: Dict[str, Any] = Body(...),
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Update a tag"""
    try:
        tag = await tags_service.update_tag(
            tag_id=tag_id,
            name=tag_data.get("name"),
            description=tag_data.get("description"),
            prompts=tag_data.get("prompts"),
            aliases=tag_data.get("aliases"),
            parent_tag_ids=tag_data.get("parent_tag_ids"),
            child_tag_ids=tag_data.get("child_tag_ids"),
            review_threshold=tag_data.get("review_threshold"),
            auto_threshold=tag_data.get("auto_threshold"),
            is_active=tag_data.get("is_active")
        )
        return tag
    except Exception as e:
        logger.error("Failed to update tag", tag_id=tag_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tags/{tag_id}/prompts")
async def get_tag_prompts(
    tag_id: str,
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get prompts for a tag"""
    try:
        tag = await tags_service.get_tag(tag_id)
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        return {"tag_id": tag_id, "prompts": tag.get("prompts", [])}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch tag prompts", tag_id=tag_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/tags/{tag_id}/prompts")
async def add_tag_prompt(
    tag_id: str,
    prompt_data: Dict[str, str] = Body(...),
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Add a prompt to a tag"""
    try:
        tag = await tags_service.get_tag(tag_id)
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        
        prompts = tag.get("prompts", [])
        new_prompt = prompt_data.get("prompt")
        if not new_prompt:
            raise HTTPException(status_code=400, detail="Prompt text is required")
        
        if new_prompt not in prompts:
            prompts.append(new_prompt)
            await tags_service.update_tag(tag_id=tag_id, prompts=prompts)
        
        return {"tag_id": tag_id, "prompts": prompts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add tag prompt", tag_id=tag_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/tags/{tag_id}/prompts/{prompt_index}")
async def update_tag_prompt(
    tag_id: str,
    prompt_index: int,
    prompt_data: Dict[str, str] = Body(...),
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Update a prompt at a specific index"""
    try:
        tag = await tags_service.get_tag(tag_id)
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        
        prompts = tag.get("prompts", [])
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise HTTPException(status_code=400, detail="Invalid prompt index")
        
        new_prompt = prompt_data.get("prompt")
        if not new_prompt:
            raise HTTPException(status_code=400, detail="Prompt text is required")
        
        prompts[prompt_index] = new_prompt
        await tags_service.update_tag(tag_id=tag_id, prompts=prompts)
        
        return {"tag_id": tag_id, "prompts": prompts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update tag prompt", tag_id=tag_id, prompt_index=prompt_index, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/tags/{tag_id}/prompts/{prompt_index}")
async def delete_tag_prompt(
    tag_id: str,
    prompt_index: int,
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Delete a prompt at a specific index"""
    try:
        tag = await tags_service.get_tag(tag_id)
        if not tag:
            raise HTTPException(status_code=404, detail="Tag not found")
        
        prompts = tag.get("prompts", [])
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise HTTPException(status_code=400, detail="Invalid prompt index")
        
        prompts.pop(prompt_index)
        await tags_service.update_tag(tag_id=tag_id, prompts=prompts)
        
        return {"tag_id": tag_id, "prompts": prompts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete tag prompt", tag_id=tag_id, prompt_index=prompt_index, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/tags/{tag_id}/prompts")
async def replace_tag_prompts(
    tag_id: str,
    prompts_data: Dict[str, List[str]] = Body(...),
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Replace all prompts for a tag"""
    try:
        prompts = prompts_data.get("prompts", [])
        await tags_service.update_tag(tag_id=tag_id, prompts=prompts)
        tag = await tags_service.get_tag(tag_id)
        return {"tag_id": tag_id, "prompts": tag.get("prompts", [])}
    except Exception as e:
        logger.error("Failed to replace tag prompts", tag_id=tag_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Blacklist API endpoints
@app.get("/blacklist")
async def get_blacklist(
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Get all blacklisted tags"""
    try:
        blacklist = await tags_service.get_blacklisted_tags()
        return blacklist
    except Exception as e:
        logger.error("Failed to fetch blacklist", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/blacklist")
async def add_to_blacklist(
    blacklist_data: Dict[str, Any] = Body(...),
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Add a tag to the blacklist"""
    try:
        tag_name = blacklist_data.get("tag_name")
        if not tag_name:
            raise HTTPException(status_code=400, detail="tag_name is required")
        
        result = await tags_service.add_to_blacklist(
            tag_name=tag_name,
            tag_id=blacklist_data.get("tag_id"),
            reason=blacklist_data.get("reason")
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add to blacklist", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/blacklist/{tag_name}")
async def remove_from_blacklist(
    tag_name: str,
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Remove a tag from the blacklist"""
    try:
        success = await tags_service.remove_from_blacklist(tag_name)
        if not success:
            raise HTTPException(status_code=404, detail="Tag not found in blacklist")
        return {"success": True, "message": f"Tag {tag_name} removed from blacklist"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove from blacklist", tag_name=tag_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/tags/sync-prompts")
async def sync_prompts_from_file(
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Sync prompts from tag_prompts.json to database"""
    try:
        result = await sync_tags_from_prompts()
        return result
    except Exception as e:
        logger.error("Failed to sync prompts from file", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/tags/export-prompts")
async def export_prompts_to_file(
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Export prompts from database to tag_prompts.json format"""
    try:
        result = await export_tags_to_prompts()
        return result
    except Exception as e:
        logger.error("Failed to export prompts to file", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/tags/import")
async def import_tags_from_json(
    file: UploadFile = File(...),
    tags_service: TagsService = Depends(get_tags_service),
    # current_user = Depends(get_current_user)  # TODO: Enable auth when ready
):
    """Import tags + prompts from JSON file (tag_prompts.json format)"""
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        # Read file content
        content = await file.read()
        try:
            tag_prompts = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
        
        if not isinstance(tag_prompts, dict):
            raise HTTPException(status_code=400, detail="JSON must be an object/dictionary")
        
        logger.info("Importing tags from JSON file", tag_count=len(tag_prompts), filename=file.filename)
        
        tags_created = 0
        tags_updated = 0
        tags_skipped = 0
        errors = []
        
        async with async_session_maker() as session:
            async with session.begin():
                for tag_name, prompts in tag_prompts.items():
                    try:
                        # Validate prompts
                        if not prompts or not isinstance(prompts, list) or len(prompts) == 0:
                            tags_skipped += 1
                            continue
                        
                        # Generate tag_id from name (same logic as tag_sync.py)
                        import re
                        tag_id = tag_name.lower()
                        tag_id = re.sub(r'[^a-z0-9_]', '_', tag_id)
                        tag_id = re.sub(r'_+', '_', tag_id)  # Replace multiple underscores with single
                        tag_id = tag_id.strip('_')
                        tag_id = f'tag_{tag_id}'
                        
                        # Check if tag exists
                        query = sa.select(tags_table).where(
                            (tags_table.c.tag_id == tag_id) | (tags_table.c.name == tag_name)
                        )
                        result = await session.execute(query)
                        existing = result.first()
                        
                        if existing:
                            # Convert SQLAlchemy Row to dict
                            if hasattr(existing, '_mapping'):
                                existing_dict = dict(existing._mapping)
                            elif hasattr(existing, '_asdict'):
                                existing_dict = existing._asdict()
                            else:
                                existing_dict = {
                                    'tag_id': getattr(existing, 'tag_id', None),
                                    'name': getattr(existing, 'name', None)
                                }
                            
                            # Update existing tag
                            update_query = sa.update(tags_table).where(
                                tags_table.c.tag_id == existing_dict['tag_id']
                            ).values(
                                name=tag_name,
                                is_active=True,
                                prompts=prompts
                            )
                            await session.execute(update_query)
                            tags_updated += 1
                        else:
                            # Create new tag
                            insert_query = sa.insert(tags_table).values(
                                tag_id=tag_id,
                                name=tag_name,
                                is_active=True,
                                prompts=prompts
                            )
                            await session.execute(insert_query)
                            tags_created += 1
                    except Exception as e:
                        logger.warning("Failed to import tag", tag_name=tag_name, error=str(e))
                        errors.append(f"{tag_name}: {str(e)}")
                        continue
        
        logger.info(
            "Tag import completed",
            tags_created=tags_created,
            tags_updated=tags_updated,
            tags_skipped=tags_skipped,
            errors_count=len(errors),
            filename=file.filename
        )
        
        return {
            "success": True,
            "tags_created": tags_created,
            "tags_updated": tags_updated,
            "tags_skipped": tags_skipped,
            "errors": errors[:10] if errors else [],  # Return first 10 errors
            "errors_count": len(errors),
            "total_in_file": len(tag_prompts)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to import tags from JSON", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/frames/{frame_id}/image")
async def get_frame_image(frame_id: str):
    """Serve frame image by frame ID"""
    try:
        # Query frame from database
        async with async_session_maker() as session:
            query = sa.select(frame_samples_table).where(frame_samples_table.c.id == frame_id)
            result = await session.execute(query)
            frame_record = result.first()
            
            if not frame_record:
                raise HTTPException(status_code=404, detail="Frame not found")
            
            # Convert Row object to dict using _mapping attribute (SQLAlchemy 2.0+)
            if hasattr(frame_record, '_mapping'):
                frame_dict = dict(frame_record._mapping)
            elif hasattr(frame_record, '_asdict'):
                frame_dict = frame_record._asdict()
            else:
                # Fallback: access columns directly
                frame_dict = {col.name: getattr(frame_record, col.name, None) for col in frame_samples_table.columns}
            
            file_path = frame_dict.get("file_path")
            job_id = frame_dict.get("job_id")
            frame_number = frame_dict.get("frame_number", 0)
            
            # Try multiple possible paths for the frame file
            possible_paths = []
            
            if file_path:
                possible_paths.append(file_path)
            
            # Try shared volume paths (mounted at /app/frames in API container)
            # The sampler stores frames at /app/.cache/frames/{job_id}/frame_{number}.jpg
            # The volume root is different in each container, so we need to check both
            if job_id:
                # Volume is mounted at /app/frames in API container
                # Sampler stores at /app/.cache/frames/{job_id}/frame_{number}.jpg
                # In the volume, this becomes {job_id}/frame_{number}.jpg
                # So in API container, it should be /app/frames/{job_id}/frame_{number}.jpg
                shared_volume_path = f"/app/frames/{job_id}/frame_{frame_number:06d}.jpg"
                possible_paths.append(shared_volume_path)
                
                # Also try the original path structure in case it's different
                alt_path = f"/app/frames/.cache/frames/{job_id}/frame_{frame_number:06d}.jpg"
                possible_paths.append(alt_path)
            
            # Note: Old frames stored before shared volume setup won't be accessible
            # They were stored in container-specific locations that aren't shared
            # Only frames stored after the shared volume setup will be accessible
            
            # Find the first existing path (optimized synchronous check)
            actual_path = None
            
            # Check paths synchronously first (much faster for local files)
            for path in possible_paths:
                if not path:
                    continue
                
                try:
                    # Quick synchronous check (faster than async for local files)
                    if os.path.exists(path) and os.path.isfile(path) and os.access(path, os.R_OK):
                        actual_path = path
                        break
                except (OSError, PermissionError) as e:
                    logger.debug(f"Error checking path {path}: {e}")
                    continue
            
            # If still not found, try async check as fallback (for network mounts)
            # but only if synchronous check didn't find anything
            if not actual_path:
                loop = asyncio.get_event_loop()
                for path in possible_paths:
                    if not path:
                        continue
                    
                    try:
                        # Use executor for async file check (for network mounts)
                        def check_path():
                            return os.path.exists(path) and os.path.isfile(path) and os.access(path, os.R_OK)
                        
                        exists_and_readable = await asyncio.wait_for(
                            loop.run_in_executor(None, check_path),
                            timeout=1.0  # 1 second timeout per path check
                        )
                        if exists_and_readable:
                            actual_path = path
                            break
                    except asyncio.TimeoutError:
                        logger.debug(f"Timeout checking path: {path}")
                        continue
                    except Exception as e:
                        logger.debug(f"Error checking path {path}: {e}")
                        continue
            
            if not actual_path:
                logger.warning(
                    "Frame file not found",
                    frame_id=frame_id,
                    job_id=job_id,
                    frame_number=frame_number,
                    tried_paths=possible_paths
                )
                raise HTTPException(status_code=404, detail="Frame image file not found")
            
            # Read and return image file
            return FileResponse(
                actual_path,
                media_type="image/jpeg",
                filename=f"frame_{frame_id}.jpg"
            )
            
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Timeout serving frame image", frame_id=frame_id)
        raise HTTPException(status_code=504, detail="Timeout loading frame image")
    except Exception as e:
        logger.error("Failed to serve frame image", frame_id=frame_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load frame image")


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests"""
    from fastapi.responses import Response
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
