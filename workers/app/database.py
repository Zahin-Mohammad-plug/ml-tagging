"""Database connection for workers"""

import os
import asyncpg
import asyncio
from typing import Optional, Dict, Any, List
import structlog
from contextlib import asynccontextmanager

from .config import get_worker_settings

logger = structlog.get_logger()
settings = get_worker_settings()


def run_async(coro):
    """
    Safely run async function in Celery/Docker environment.
    Handles cases where event loop may or may not already exist.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            # Create a new event loop in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            # Loop exists but not running, use it
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)

class DatabaseConnection:
    """Async database connection manager for workers"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        # Convert SQLAlchemy URL format to asyncpg format
        db_url = settings.database_url
        if db_url.startswith('postgresql+asyncpg://'):
            db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
        self.database_url = db_url
    
    async def connect(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch single record"""
        async with self.get_connection() as conn:
            record = await conn.fetchrow(query, *args)
            return dict(record) if record else None
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch all records"""
        async with self.get_connection() as conn:
            records = await conn.fetch(query, *args)
            return [dict(record) for record in records]

# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None

def get_database_connection() -> DatabaseConnection:
    """Get global database connection instance"""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection

# Database helper functions for workers
async def store_frame_samples(job_id: str, scene_id: str, frame_data: List[Dict[str, Any]]) -> List[str]:
    """Store frame sample metadata"""
    
    db = get_database_connection()
    frame_ids = []
    
    for frame_info in frame_data:
        frame_id = frame_info.get("id")  # Should be generated upstream
        
        query = """
            INSERT INTO frame_samples (id, job_id, scene_id, frame_number, timestamp_seconds, file_path, width, height)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        await db.execute(
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
        
        frame_ids.append(frame_id)
    
    return frame_ids

async def store_embeddings(frame_embeddings: List[Dict[str, Any]]) -> None:
    """Store frame embeddings"""
    
    db = get_database_connection()
    
    for embedding_data in frame_embeddings:
        query = """
            INSERT INTO embeddings (id, frame_id, model_name, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5)
        """
        
        await db.execute(
            query,
            embedding_data["id"],
            embedding_data["frame_id"],
            embedding_data["model_name"],
            embedding_data["embedding"],
            embedding_data.get("metadata", {})
        )

async def store_text_analysis(text_results: List[Dict[str, Any]]) -> None:
    """Store ASR/OCR results"""
    
    db = get_database_connection()
    
    for text_data in text_results:
        query = """
            INSERT INTO text_analysis (
                id, frame_id, job_id, analysis_type, text_content, 
                confidence, language, start_time, end_time, bounding_box, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        await db.execute(
            query,
            text_data["id"],
            text_data.get("frame_id"),
            text_data["job_id"],
            text_data["analysis_type"],
            text_data["text_content"],
            text_data.get("confidence"),
            text_data.get("language"),
            text_data.get("start_time"),
            text_data.get("end_time"),
            text_data.get("bounding_box"),
            text_data.get("metadata", {})
        )

async def get_job_frames(job_id: str) -> List[Dict[str, Any]]:
    """Get all frames for a job"""
    
    db = get_database_connection()
    
    query = """
        SELECT id, job_id, scene_id, frame_number, timestamp_seconds, file_path, width, height
        FROM frame_samples
        WHERE job_id = $1
        ORDER BY frame_number
    """
    
    return await db.fetch_all(query, job_id)

async def get_frame_embeddings(frame_ids: List[str], model_name: str) -> List[Dict[str, Any]]:
    """Get embeddings for frames"""
    
    db = get_database_connection()
    
    query = """
        SELECT e.*, f.frame_number, f.timestamp_seconds
        FROM embeddings e
        JOIN frame_samples f ON e.frame_id = f.id
        WHERE e.frame_id = ANY($1) AND e.model_name = $2
        ORDER BY f.frame_number
    """
    
    return await db.fetch_all(query, frame_ids, model_name)

async def get_job_metadata(job_id: str) -> Dict[str, Any]:
    """Get job metadata including job options"""
    
    db = get_database_connection()
    
    query = """
        SELECT metadata FROM jobs WHERE job_id = $1
    """
    
    result = await db.fetch_one(query, job_id)
    if result and isinstance(result, dict):
        metadata_raw = result.get("metadata")
        if metadata_raw:
            # Handle JSONB column - it might be a dict, string, or already parsed
            if isinstance(metadata_raw, dict):
                return metadata_raw
            elif isinstance(metadata_raw, str):
                try:
                    import json
                    return json.loads(metadata_raw)
                except (json.JSONDecodeError, TypeError):
                    return {}
            else:
                # Try to convert to dict if possible
                return dict(metadata_raw) if hasattr(metadata_raw, '__dict__') else {}
    return {}

async def get_job_text_analysis(job_id: str, analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get text analysis results for a job"""
    
    db = get_database_connection()
    
    if analysis_type:
        query = """
            SELECT * FROM text_analysis
            WHERE job_id = $1 AND analysis_type = $2
            ORDER BY start_time NULLS LAST
        """
        return await db.fetch_all(query, job_id, analysis_type)
    else:
        query = """
            SELECT * FROM text_analysis
            WHERE job_id = $1
            ORDER BY analysis_type, start_time NULLS LAST
        """
        return await db.fetch_all(query, job_id)

async def get_active_tags() -> List[Dict[str, Any]]:
    """Get all active tags for similarity matching"""
    
    db = get_database_connection()
    
    query = """
        SELECT tag_id, name, synonyms, embedding, review_threshold, auto_threshold,
               parent_tag_ids, child_tag_ids
        FROM tags
        WHERE is_active = true
        ORDER BY name
    """
    
    return await db.fetch_all(query)

async def store_suggestions(suggestions: List[Dict[str, Any]]) -> List[str]:
    """Store ML suggestions"""
    
    db = get_database_connection()
    suggestion_ids = []
    
    for suggestion in suggestions:
        query = """
            INSERT INTO suggestions (
                id, job_id, scene_id, tag_id, tag_name, confidence,
                vision_confidence, asr_confidence, ocr_confidence, temporal_consistency,
                calibrated_confidence, evidence_frames, reasoning, signal_details, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """
        
        await db.execute(
            query,
            suggestion["id"],
            suggestion["job_id"],
            suggestion["scene_id"], 
            suggestion["tag_id"],
            suggestion["tag_name"],
            suggestion["confidence"],
            suggestion.get("vision_confidence"),
            suggestion.get("asr_confidence"),
            suggestion.get("ocr_confidence"),
            suggestion.get("temporal_consistency"),
            suggestion.get("calibrated_confidence"),
            suggestion.get("evidence_frames", []),
            suggestion.get("reasoning"),
            suggestion.get("signal_details", {}),
            suggestion.get("status", "pending")
        )
        
        suggestion_ids.append(suggestion["id"])
    
    return suggestion_ids