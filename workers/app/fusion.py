"""
Fusion Worker - Multi-modal fusion and tag suggestion generation

This worker combines signals from multiple modalities (vision, audio, text)
to generate calibrated tag suggestions with evidence.
"""

import uuid
import json as json_module
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict
from celery import Task
import structlog
import numpy as np

from .celery_app import app
from .database import (
    get_database_connection,
    get_job_frames,
    get_active_tags,
    store_suggestions
)
from .config import get_worker_settings

logger = structlog.get_logger(__name__)


class FusionTask(Task):
    """Base task for fusion worker"""
    
    _db = None
    _settings = None
    
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


@app.task(bind=True, base=FusionTask, name='app.fusion.generate_suggestions')
def generate_suggestions(
    self,
    job_id: str,
    embeddings_result: Dict[str, Any],
    text_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate tag suggestions by fusing multi-modal signals
    
    Args:
        job_id: Unique job identifier
        embeddings_result: Result from embeddings worker
        text_result: Result from ASR/OCR worker
    
    Returns:
        {
            "success": bool,
            "suggestion_count": int,
            "high_confidence_count": int,
            "medium_confidence_count": int,
            "low_confidence_count": int
        }
    """
    
    logger.info(
        "Starting fusion and suggestion generation",
        job_id=job_id,
        embeddings_count=embeddings_result.get("embedding_count", 0),
        text_entries=text_result.get("total_text_entries", 0)
    )
    
    try:
        # Use direct asyncpg connections to avoid pool conflicts
        import asyncio
        import asyncpg
        from .config import get_worker_settings
        
        settings = get_worker_settings()
        # Convert SQLAlchemy URL to asyncpg format
        db_url = settings.database_url
        if db_url.startswith('postgresql+asyncpg://'):
            db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
        
        # Get model name from database settings (with fallback to config/env)
        model_name = getattr(settings, 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        try:
            # Query database for clip_model_name setting
            import asyncpg
            async def get_model_name():
                conn = await asyncpg.connect(db_url)
                try:
                    row = await conn.fetchrow(
                        "SELECT value FROM settings WHERE key = $1 AND user_id IS NULL",
                        "clip_model_name"
                    )
                    if row and row['value']:
                        value = row['value']
                        # Handle JSONB values - they might be stored as JSON strings
                        if isinstance(value, str):
                            # Try to parse as JSON if it looks like a JSON string
                            if value.startswith('"') and value.endswith('"'):
                                import json
                                try:
                                    value = json.loads(value)
                                except:
                                    pass
                        return str(value) if value else None
                finally:
                    await conn.close()
                return None
            
            from .database import run_async
            db_model_name = run_async(get_model_name())
            if db_model_name:
                model_name = db_model_name
                logger.debug(f"Using clip_model_name from database: {model_name}")
        except Exception as e:
            logger.debug(f"Could not read clip_model_name from database, using config default: {e}")
        
        async def run_fusion_pipeline():
            """Run complete fusion pipeline with separate connections"""
            # Get device setting inside the async function to avoid scoping issues
            vision_device = getattr(settings, 'vision_device', 'auto')
            # Convert 'auto' to None so CLIPEmbedder can auto-detect (preferring GPU)
            device = None if vision_device == 'auto' else vision_device
            # Import json here to avoid scoping issues
            import json as json_module
            # Create separate connections for each operation
            conn1 = await asyncpg.connect(db_url)
            conn2 = await asyncpg.connect(db_url)
            
            try:
                # Get frames with embeddings
                # Check if we have reusable frames (frame_ids passed directly)
                job_metadata = await conn1.fetchval(
                    "SELECT metadata FROM jobs WHERE job_id = $1", job_id
                )
                reusable_frames = None
                if job_metadata:
                    import json
                    metadata = json.loads(job_metadata) if isinstance(job_metadata, str) else job_metadata
                    reusable_frames = metadata.get("reusable_frames")
                
                # Check for reusable frames in metadata OR embeddings_result (fallback)
                frame_ids_to_use = None
                if reusable_frames and reusable_frames.get("frame_ids"):
                    frame_ids_to_use = reusable_frames.get("frame_ids", [])
                elif embeddings_result.get("reused") and embeddings_result.get("frame_ids"):
                    # Fallback: check embeddings_result for frame_ids when reusing
                    frame_ids_to_use = embeddings_result.get("frame_ids", [])
                    logger.info(
                        "Using frame_ids from embeddings_result (reused frames)",
                        job_id=job_id,
                        frame_count=len(frame_ids_to_use) if frame_ids_to_use else 0
                    )
                
                if frame_ids_to_use:
                    # Query by frame_ids for reusable frames
                    frames_query = """
                        SELECT 
                            f.id, f.job_id, f.scene_id, f.frame_number, 
                            f.timestamp_seconds, f.file_path, f.width, f.height,
                            CASE WHEN e.embedding IS NOT NULL THEN e.embedding::text ELSE NULL END as embedding
                        FROM frame_samples f
                        LEFT JOIN embeddings e ON f.id = e.frame_id AND e.model_name = $2
                        WHERE f.id = ANY($1::text[])
                        ORDER BY f.frame_number
                    """
                    frames_records = await conn1.fetch(frames_query, frame_ids_to_use, model_name)
                    logger.info(
                        "Using reusable frames",
                        job_id=job_id,
                        frame_count=len(frame_ids_to_use),
                        original_job_id=reusable_frames.get("original_job_id") if reusable_frames else None
                    )
                else:
                    # Normal query by job_id
                    frames_query = """
                        SELECT 
                            f.id, f.job_id, f.scene_id, f.frame_number, 
                            f.timestamp_seconds, f.file_path, f.width, f.height,
                            CASE WHEN e.embedding IS NOT NULL THEN e.embedding::text ELSE NULL END as embedding
                        FROM frame_samples f
                        LEFT JOIN embeddings e ON f.id = e.frame_id AND e.model_name = $2
                        WHERE f.job_id = $1
                        ORDER BY f.frame_number
                    """
                    frames_records = await conn1.fetch(frames_query, job_id, model_name)
                logger.info(f"[DEBUG] Fetched {len(frames_records)} frame records from database", job_id=job_id)
                
                frames = []
                embedding_parse_errors = 0
                embeddings_parsed = 0
                embeddings_null = 0
                
                for i, record in enumerate(frames_records):
                    frame_dict = dict(record)
                    # Convert pgvector text to list - handle JSON array format
                    embedding = frame_dict.get('embedding')
                    
                    # Debug: Log first few embeddings
                    if i < 3:
                        logger.debug(
                            f"[DEBUG] Frame {i} embedding check",
                            job_id=job_id,
                            frame_id=frame_dict.get('id'),
                            frame_number=frame_dict.get('frame_number'),
                            embedding_is_none=embedding is None,
                            embedding_type=type(embedding).__name__ if embedding is not None else 'None',
                            embedding_preview=str(embedding)[:50] if embedding else 'None'
                        )
                    
                    if embedding is not None and embedding != '':
                        try:
                            # Embedding is cast to text, so it should be a string in JSON array format
                            if isinstance(embedding, str):
                                # Parse JSON array: "[-0.02205164,0.049008124,...]"
                                frame_dict['embedding'] = json_module.loads(embedding)
                                embeddings_parsed += 1
                                if i < 3:
                                    logger.debug(
                                        f"[DEBUG] Successfully parsed embedding for frame {i}",
                                        job_id=job_id,
                                        frame_id=frame_dict.get('id'),
                                        embedding_length=len(frame_dict['embedding']) if frame_dict['embedding'] else 0
                                    )
                            elif isinstance(embedding, (list, tuple)):
                                # Already a list (shouldn't happen with text cast, but handle it)
                                frame_dict['embedding'] = list(embedding)
                                embeddings_parsed += 1
                            else:
                                # Unknown type
                                logger.warning(
                                    "[DEBUG] Unexpected embedding type",
                                    job_id=job_id,
                                    frame_id=frame_dict.get('id'),
                                    embedding_type=type(embedding).__name__,
                                    embedding_value=str(embedding)[:100]
                                )
                                frame_dict['embedding'] = None
                                embedding_parse_errors += 1
                        except json_module.JSONDecodeError as e:
                            logger.warning(
                                "[DEBUG] Failed to parse embedding as JSON",
                                job_id=job_id,
                                frame_id=frame_dict.get('id'),
                                frame_number=frame_dict.get('frame_number'),
                                embedding_preview=embedding[:100] if embedding else 'None',
                                embedding_length=len(embedding) if embedding else 0,
                                error=str(e)
                            )
                            frame_dict['embedding'] = None
                            embedding_parse_errors += 1
                        except Exception as e:
                            logger.warning(
                                "[DEBUG] Failed to parse embedding - unexpected error",
                                job_id=job_id,
                                frame_id=frame_dict.get('id'),
                                frame_number=frame_dict.get('frame_number'),
                                embedding_type=type(embedding).__name__,
                                error=str(e),
                                error_type=type(e).__name__
                            )
                            frame_dict['embedding'] = None
                            embedding_parse_errors += 1
                    else:
                        # No embedding (NULL or empty string)
                        frame_dict['embedding'] = None
                        embeddings_null += 1
                        if i < 3:
                            logger.debug(
                                f"[DEBUG] Frame {i} has no embedding (NULL or empty)",
                                job_id=job_id,
                                frame_id=frame_dict.get('id'),
                                frame_number=frame_dict.get('frame_number')
                            )
                    frames.append(frame_dict)
                
                logger.info(
                    "[DEBUG] Frame embedding parsing summary",
                    job_id=job_id,
                    total_frames=len(frames),
                    embeddings_parsed=embeddings_parsed,
                    embeddings_null=embeddings_null,
                    parse_errors=embedding_parse_errors,
                    frames_with_embeddings=sum(1 for f in frames if f.get('embedding') is not None)
                )
                
                if embedding_parse_errors > 0:
                    logger.warning(
                        "[DEBUG] Some embeddings failed to parse",
                        job_id=job_id,
                        parse_errors=embedding_parse_errors,
                        total_frames=len(frames)
                    )
                
                # Load text analysis results (ASR/OCR) and associate with frames
                # Note: text_analysis table doesn't have timestamp_seconds column, use start_time for ASR
                text_analysis_query = """
                    SELECT 
                        frame_id, analysis_type, text_content, confidence,
                        start_time, end_time
                    FROM text_analysis
                    WHERE job_id = $1
                    ORDER BY start_time NULLS LAST
                """
                text_records = await conn1.fetch(text_analysis_query, job_id)
                
                # Build a map of frame_id -> text content
                frame_text_map = {}
                for text_record in text_records:
                    text_dict = dict(text_record)
                    frame_id = text_dict.get('frame_id')
                    text_content = text_dict.get('text_content', '')
                    analysis_type = text_dict.get('analysis_type', '')
                    
                    if frame_id and text_content:
                        if frame_id not in frame_text_map:
                            frame_text_map[frame_id] = []
                        frame_text_map[frame_id].append(text_content)
                
                # Also match ASR segments to frames by timestamp (ASR might not have frame_id)
                # Match ASR segments to nearby frames (within 2 seconds)
                for text_record in text_records:
                    text_dict = dict(text_record)
                    analysis_type = text_dict.get('analysis_type', '')
                    frame_id = text_dict.get('frame_id')
                    start_time = text_dict.get('start_time')
                    text_content = text_dict.get('text_content', '')
                    
                    # If ASR segment doesn't have frame_id, match by timestamp
                    if analysis_type == 'asr' and not frame_id and start_time is not None and text_content:
                        # Find frames within 2 seconds of this ASR segment
                        for frame in frames:
                            frame_time = frame.get('timestamp_seconds', 0)
                            if frame_time is not None and abs(frame_time - start_time) <= 2.0:
                                frame_id_for_match = frame.get('id')
                                if frame_id_for_match:
                                    if frame_id_for_match not in frame_text_map:
                                        frame_text_map[frame_id_for_match] = []
                                    frame_text_map[frame_id_for_match].append(text_content)
                
                # Add text content to frames
                for frame in frames:
                    frame_id = frame.get('id')
                    if frame_id in frame_text_map:
                        # Combine all text content for this frame
                        frame['text_content'] = ' '.join(frame_text_map[frame_id])
                    else:
                        frame['text_content'] = ''
                
                # Get active tags with embeddings and prompts
                tags_query = """
                    SELECT tag_id, name, description, synonyms, parent_tag_ids, child_tag_ids, embedding::text as embedding, prompts
                    FROM tags
                    WHERE is_active = TRUE
                    ORDER BY name
                """
                tags_records = await conn2.fetch(tags_query)
                total_active_tags_in_db = len(tags_records)
                logger.info(
                    "Loaded active tags from database",
                    job_id=job_id,
                    total_active_tags=total_active_tags_in_db
                )
                active_tags = []
                tags_needing_embeddings = []
                
                # Load tag prompts from file as backup (in case database prompts are empty)
                import os
                tag_prompts_file = os.environ.get("TAG_PROMPTS_PATH", "/app/prompts/tag_prompts.json")
                file_tag_prompts = {}
                if os.path.exists(tag_prompts_file):
                    try:
                        with open(tag_prompts_file, 'r', encoding='utf-8') as f:
                            file_tag_prompts = json_module.load(f)
                        logger.debug(f"Loaded {len(file_tag_prompts)} tag prompts from file", job_id=job_id)
                    except Exception as e:
                        logger.warning(f"Failed to load tag_prompts.json: {e}", job_id=job_id)
                        pass
                else:
                    logger.warning(f"tag_prompts.json not found at {tag_prompts_file}", job_id=job_id)
                
                for record in tags_records:
                    tag_dict = dict(record)
                    # Convert pgvector to list/array - handle multiple formats (same as frames)
                    embedding = tag_dict.get('embedding')
                    if embedding is not None:
                        try:
                            # Case 1: Already a list/array
                            if isinstance(embedding, (list, tuple)):
                                tag_dict['embedding'] = list(embedding)
                            # Case 2: String format
                            elif isinstance(embedding, str):
                                try:
                                    tag_dict['embedding'] = json_module.loads(embedding)
                                except json_module.JSONDecodeError:
                                    embedding_str = embedding.strip('[]')
                                    tag_dict['embedding'] = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                            # Case 3: numpy array
                            elif hasattr(embedding, 'tolist'):
                                tag_dict['embedding'] = embedding.tolist()
                            else:
                                tag_dict['embedding'] = list(embedding) if hasattr(embedding, '__iter__') else None
                        except Exception as e:
                            logger.warning(
                                "Failed to parse tag embedding",
                                job_id=job_id,
                                tag_id=tag_dict.get('tag_id'),
                                tag_name=tag_dict.get('name'),
                                embedding_type=type(embedding).__name__,
                                error=str(e)
                            )
                            tag_dict['embedding'] = None
                    
                    # Parse prompts (JSONB)
                    prompts = tag_dict.get('prompts', [])
                    if isinstance(prompts, str):
                        try:
                            prompts = json_module.loads(prompts)
                        except:
                            prompts = []
                    
                    # If no prompts in database, try to get from file
                    if not prompts or len(prompts) == 0:
                        tag_name = tag_dict.get('name', '')
                        # Try case-insensitive match
                        for key in file_tag_prompts.keys():
                            if key.lower() == tag_name.lower():
                                prompts = file_tag_prompts[key]
                                break
                    
                    tag_dict['prompts'] = prompts if prompts else [tag_dict.get('name', '')]
                    
                    # Track tags that need embeddings or have mismatched dimensions
                    # Check if existing embedding dimension matches current model
                    tag_embedding = tag_dict.get('embedding')
                    if tag_embedding:
                        tag_emb_dim = len(tag_embedding) if isinstance(tag_embedding, (list, np.ndarray)) else 0
                        # Get expected dimension from embedder (will be loaded later)
                        # For now, we'll check during embedding generation
                        # If dimension mismatch, regenerate
                        if tag_emb_dim > 0:
                            # We'll validate dimension later when we have the embedder
                            pass
                    
                    if not tag_dict.get('embedding'):
                        tags_needing_embeddings.append(tag_dict)
                    
                    active_tags.append(tag_dict)
                
                # Load CLIP embedder to check dimensions and generate missing embeddings
                embedder_for_tags = None
                expected_embedding_dim = None
                try:
                    from .embeddings import CLIPEmbedder
                    embedder_for_tags = CLIPEmbedder(model_name=model_name, device=device)
                    expected_embedding_dim = embedder_for_tags.embedding_dim
                    logger.info(
                        f"Loaded embedder for tag generation, expected_dim={expected_embedding_dim}",
                        job_id=job_id,
                        model_name=model_name
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load CLIP embedder for tag embedding validation",
                        job_id=job_id,
                        error=str(e)
                    )
                
                # Check existing tag embeddings for dimension mismatches
                if embedder_for_tags and expected_embedding_dim:
                    for tag_dict in active_tags:
                        tag_embedding = tag_dict.get('embedding')
                        if tag_embedding:
                            tag_emb_dim = len(tag_embedding) if isinstance(tag_embedding, (list, np.ndarray)) else 0
                            if tag_emb_dim > 0 and tag_emb_dim != expected_embedding_dim:
                                logger.info(
                                    f"Tag embedding dimension mismatch, will regenerate",
                                    job_id=job_id,
                                    tag_name=tag_dict.get('name'),
                                    current_dim=tag_emb_dim,
                                    expected_dim=expected_embedding_dim,
                                    model_name=model_name
                                )
                                # Mark for regeneration
                                tag_dict['embedding'] = None
                                if tag_dict not in tags_needing_embeddings:
                                    tags_needing_embeddings.append(tag_dict)
                
                # Generate embeddings for tags that are missing them or have mismatched dimensions
                # Note: This requires CLIP model loading which may fail due to cache permissions
                # If it fails, we'll just skip tags without embeddings and continue with existing ones
                if tags_needing_embeddings:
                    logger.info(
                        "Attempting to generate embeddings for tags missing them",
                        job_id=job_id,
                        tag_count=len(tags_needing_embeddings)
                    )
                    
                    try:
                        # Import CLIP embedder to generate embeddings
                        from .embeddings import CLIPEmbedder
                        
                        # Try to load CLIP model - may fail due to cache permissions
                        # Use embedder_for_tags if already loaded, otherwise create new one
                        try:
                            if embedder_for_tags:
                                embedder = embedder_for_tags
                            else:
                                embedder = CLIPEmbedder(model_name=model_name, device=device)
                        except (PermissionError, OSError) as e:
                            logger.warning(
                                "Failed to load CLIP model for tag embedding generation",
                                job_id=job_id,
                                error=str(e),
                                message="Skipping tag embedding generation. Tags without embeddings will be ignored."
                            )
                            # Remove tags without embeddings from active_tags
                            active_tags = [t for t in active_tags if t.get('embedding')]
                            tags_needing_embeddings = []
                        else:
                            # Use the file_tag_prompts we already loaded earlier (line 245)
                            # No need to reload - just use file_tag_prompts
                            logger.info(
                                f"[DEBUG] Generating embeddings for {len(tags_needing_embeddings)} tags",
                                job_id=job_id,
                                file_prompts_loaded=len(file_tag_prompts),
                                sample_tag_names=[t.get('name') for t in tags_needing_embeddings[:5]]
                            )
                            
                            # Generate and store embeddings for each tag
                            for tag_dict in tags_needing_embeddings:
                                try:
                                    tag_id = tag_dict["tag_id"]
                                    tag_name = tag_dict["name"]
                                    
                                    # Build enhanced prompt using name, description, aliases, and existing prompts
                                    # This creates a richer embedding that better captures tag semantics
                                    prompt_parts = []
                                    
                                    # Start with tag name
                                    prompt_parts.append(tag_name)
                                    
                                    # Add description if available
                                    description = tag_dict.get('description')
                                    if description and isinstance(description, str) and description.strip():
                                        prompt_parts.append(description.strip())
                                    
                                    # Add aliases/synonyms if available
                                    synonyms = tag_dict.get('synonyms', [])
                                    if synonyms:
                                        if isinstance(synonyms, str):
                                            try:
                                                synonyms = json_module.loads(synonyms)
                                            except:
                                                synonyms = []
                                        if isinstance(synonyms, list) and len(synonyms) > 0:
                                            # Add first few synonyms (limit to avoid too long prompts)
                                            synonym_text = ", ".join(synonyms[:3])
                                            prompt_parts.append(f"also known as: {synonym_text}")
                                    
                                    # Add existing prompts if available (prefer these as they're curated)
                                    prompts = tag_dict.get('prompts', [])
                                    if prompts and len(prompts) > 0:
                                        if isinstance(prompts, str):
                                            try:
                                                prompts = json_module.loads(prompts)
                                            except:
                                                prompts = []
                                        if isinstance(prompts, list) and len(prompts) > 0:
                                            # Use first prompt (most relevant) and append to parts
                                            first_prompt = prompts[0]
                                            if first_prompt and first_prompt.strip() and first_prompt.lower() != tag_name.lower():
                                                prompt_parts.append(first_prompt.strip())
                                    else:
                                        # Fallback: Try case-insensitive match in file_tag_prompts
                                        tag_prompt_key = None
                                        for key in file_tag_prompts.keys():
                                            if key.lower() == tag_name.lower():
                                                tag_prompt_key = key
                                                break
                                        
                                        if tag_prompt_key and file_tag_prompts[tag_prompt_key]:
                                            first_prompt = file_tag_prompts[tag_prompt_key][0]
                                            if first_prompt and first_prompt.strip() and first_prompt.lower() != tag_name.lower():
                                                prompt_parts.append(first_prompt.strip())
                                    
                                    # Combine all parts into a single text
                                    text = ". ".join(prompt_parts)
                                    
                                    logger.debug(
                                        "[DEBUG] Generated enhanced tag prompt",
                                        job_id=job_id,
                                        tag_name=tag_name,
                                        prompt_length=len(text),
                                        has_description=bool(description),
                                        has_synonyms=bool(synonyms),
                                        prompt_preview=text[:100] + "..." if len(text) > 100 else text
                                    )
                                    
                                    # Generate embedding
                                    embedding = embedder.encode_text(text)
                                    embedding_list = embedding.tolist()
                                    
                                    # Convert to pgvector string format
                                    embedding_str = '[' + ','.join(str(float(x)) for x in embedding_list) + ']'
                                    
                                    # Update tag in database
                                    update_query = """
                                        UPDATE tags
                                        SET embedding = $1::vector
                                        WHERE tag_id = $2
                                    """
                                    await conn2.execute(update_query, embedding_str, tag_id)
                                    
                                    # Update tag_dict for use in fusion
                                    tag_dict['embedding'] = embedding_list
                                    
                                    logger.debug(
                                        "Generated and stored tag embedding",
                                        job_id=job_id,
                                        tag_id=tag_id,
                                        tag_name=tag_name
                                    )
                                except Exception as tag_error:
                                    logger.warning(
                                        "Failed to generate embedding for tag",
                                        job_id=job_id,
                                        tag_id=tag_dict.get("tag_id"),
                                        tag_name=tag_dict.get("name"),
                                        error=str(tag_error)
                                    )
                                    # Remove this tag from active_tags if embedding generation failed
                                    active_tags = [t for t in active_tags if t.get('tag_id') != tag_dict.get('tag_id') or t.get('embedding')]
                    except Exception as e:
                        logger.warning(
                            "Tag embedding generation failed",
                            job_id=job_id,
                            error=str(e),
                            message="Continuing with tags that already have embeddings"
                        )
                        # Remove tags without embeddings from active_tags
                        active_tags = [t for t in active_tags if t.get('embedding')]
                
                if not frames:
                    logger.warning("No frames found for fusion", job_id=job_id)
                    return {
                        "success": True,
                        "suggestion_count": 0,
                        "high_confidence_count": 0,
                        "medium_confidence_count": 0,
                        "low_confidence_count": 0
                    }
                
                # Check how many frames have embeddings
                frames_with_embeddings = sum(1 for f in frames if f.get('embedding') is not None)
                
                logger.info(
                    "[DEBUG] Final frame embedding check before fusion",
                    job_id=job_id,
                    total_frames=len(frames),
                    frames_with_embeddings=frames_with_embeddings,
                    model_name=model_name,
                    sample_frame_embeddings=[f.get('embedding') is not None for f in frames[:5]]
                )
                
                # Debug logging for embedding issues
                if frames_with_embeddings == 0:
                    # Log sample of frames to debug
                    sample_frames = frames[:3] if frames else []
                    logger.error(
                        "No frame embeddings found for fusion",
                        job_id=job_id,
                        total_frames=len(frames),
                        sample_frame_ids=[f.get('id') for f in sample_frames],
                        sample_embedding_types=[type(f.get('embedding')).__name__ if f.get('embedding') is not None else 'None' for f in sample_frames],
                        sample_embedding_values=[str(f.get('embedding'))[:100] if f.get('embedding') is not None else 'None' for f in sample_frames],
                        model_name=model_name,
                        message="Frame embeddings must be generated before fusion can run. Check embeddings worker logs and database."
                    )
                    
                    # Also check if embeddings exist in database
                    check_query = """
                        SELECT COUNT(*) as count
                        FROM embeddings e
                        JOIN frame_samples f ON e.frame_id = f.id
                        WHERE f.job_id = $1 AND e.model_name = $2
                    """
                    embedding_count = await conn1.fetchval(check_query, job_id, model_name)
                    logger.error(
                        "Database embedding check",
                        job_id=job_id,
                        embeddings_in_db=embedding_count,
                        model_name=model_name,
                        message="If embeddings exist in DB but not in query results, there may be a JOIN or type conversion issue."
                    )
                    
                    return {
                        "success": False,
                        "error": f"No frame embeddings found. Embeddings worker may have failed. Found {embedding_count} embeddings in database for model {model_name}.",
                        "suggestion_count": 0,
                        "high_confidence_count": 0,
                        "medium_confidence_count": 0,
                        "low_confidence_count": 0
                    }
                
                if not active_tags:
                    logger.warning("No active tags found for matching", job_id=job_id)
                    return {
                        "success": True,
                        "suggestion_count": 0,
                        "high_confidence_count": 0,
                        "medium_confidence_count": 0,
                        "low_confidence_count": 0
                    }
                
                logger.info(
                    "Retrieved data for fusion",
                    job_id=job_id,
                    frame_count=len(frames),
                    frames_with_embeddings=frames_with_embeddings,
                    tag_count=len(active_tags),
                    tags_with_embeddings=sum(1 for t in active_tags if t.get('embedding'))
                )
                
                # Load CLIP embedder for multi-prompt matching
                # This is critical for accuracy - we need to compute similarity against all prompts
                embedder = None
                try:
                    from .embeddings import CLIPEmbedder
                    embedder = CLIPEmbedder(model_name=model_name, device=device)
                    
                    # Verify embedder is working by testing encoding
                    try:
                        test_text = "test prompt"
                        test_embedding = embedder.encode_text(test_text)
                        embedding_dim = len(test_embedding)
                        device = embedder.device
                        
                        logger.info(
                            "Loaded CLIP embedder for multi-prompt matching",
                            job_id=job_id,
                            model_name=model_name,
                            device=device,
                            embedding_dim=embedding_dim,
                            test_encoding_successful=True
                        )
                    except Exception as test_error:
                        logger.warning(
                            "CLIP embedder loaded but test encoding failed",
                            job_id=job_id,
                            model_name=model_name,
                            test_error=str(test_error),
                            message="Embedder may not be fully functional"
                        )
                except Exception as e:
                    logger.error(
                        "CRITICAL: Failed to load CLIP embedder for multi-prompt matching",
                        job_id=job_id,
                        error=str(e),
                        message="Accuracy will be reduced - only stored embeddings will be used. Check GPU/CPU availability and model cache permissions."
                    )
                    # Continue with stored embeddings, but accuracy will be reduced
                
                # Get scene_id from frames for progress tracking
                scene_id = frames[0].get('scene_id') if frames else None
                
                total_tag_count = len(active_tags)
                logger.info(
                    "Processing all active tags",
                    job_id=job_id,
                    scene_id=scene_id,
                    total_tags=total_tag_count,
                )
                
                # Update fusion progress to show we're starting tag scoring
                await _update_fusion_progress(conn1, job_id, {
                    "status": "processing",
                    "current_step": "Calculating tag scores",
                    "tags_total": total_tag_count,
                    "tags_analyzed": 0,
                    "current_tag": None,
                    "tag_progress": f"0/{total_tag_count}",
                })
                
                # Compute multi-modal scores for each tag (using all prompts)
                tag_scores = await _compute_tag_scores(self, job_id, frames, active_tags, embedder=embedder, scene_id=scene_id, conn=conn1)
                
                # Log warning if embedder failed and tags have multiple prompts
                if embedder is None:
                    tags_with_multiple_prompts = sum(1 for t in active_tags if len(t.get('prompts', [])) > 1)
                    if tags_with_multiple_prompts > 0:
                        logger.warning(
                            "Tags with multiple prompts will use fallback embeddings (reduced accuracy)",
                            job_id=job_id,
                            tags_affected=tags_with_multiple_prompts
                        )
                
                # Log top tags by raw CLIP score and final score
                if tag_scores:
                    # Sort by final score
                    sorted_by_final = sorted(tag_scores.items(), key=lambda x: x[1]["score"], reverse=True)
                    top_by_final = [
                        {
                            "tag_name": data["tag_name"],
                            "final_score": round(data["score"], 4),
                            "raw_clip_mean": round(data.get("raw_clip_stats", {}).get("mean", 0), 4),
                            "normalized_visual_mean": round(data.get("normalized_visual_stats", {}).get("mean", 0), 4),
                            "combined_score": round(data.get("combined_score", 0), 4)
                        }
                        for tag_id, data in sorted_by_final[:10]
                    ]
                    
                    # Sort by raw CLIP mean score
                    sorted_by_raw = sorted(
                        tag_scores.items(), 
                        key=lambda x: x[1].get("raw_clip_stats", {}).get("mean", 0), 
                        reverse=True
                    )
                    top_by_raw = [
                        {
                            "tag_name": data["tag_name"],
                            "raw_clip_mean": round(data.get("raw_clip_stats", {}).get("mean", 0), 4),
                            "raw_clip_max": round(data.get("raw_clip_stats", {}).get("max", 0), 4),
                            "final_score": round(data["score"], 4),
                            "normalized_visual_mean": round(data.get("normalized_visual_stats", {}).get("mean", 0), 4)
                        }
                        for tag_id, data in sorted_by_raw[:10]
                    ]
                    
                    logger.info(
                        "Computed tag scores - top by final score",
                        job_id=job_id,
                        tags_scored=len(tag_scores),
                        top_10_final=top_by_final
                    )
                    
                    logger.info(
                        "Computed tag scores - top by raw CLIP score",
                        job_id=job_id,
                        tags_scored=len(tag_scores),
                        top_10_raw=top_by_raw
                    )
                else:
                    logger.warning(
                        "No tag scores computed",
                        job_id=job_id
                    )
                
                # Build tag hierarchy map for hierarchical filtering
                tag_hierarchy = {}
                for tag in active_tags:
                    tag_id = tag.get("tag_id")
                    if tag_id:
                        tag_hierarchy[tag_id] = {
                            "parent_tag_ids": tag.get("parent_tag_ids", []),
                            "child_tag_ids": tag.get("child_tag_ids", []),
                            "name": tag.get("name", "")
                        }
                
                # Update progress to show we're generating suggestions
                await _update_fusion_progress(conn1, job_id, {
                    "status": "processing",
                    "current_step": "Generating suggestions",
                    "tags_total": len(active_tags),
                    "tags_analyzed": len(active_tags),
                    "current_tag": None,
                    "tag_progress": f"{len(active_tags)}/{len(active_tags)}"
                })
                
                # Generate suggestions from scores with hierarchical filtering
                suggestions = _generate_tag_suggestions(self, job_id, tag_scores, frames, tag_hierarchy)
                
                # Log high-confidence suggestions
                high_conf_suggestions = [s for s in suggestions if s.get("confidence", 0) >= 0.7]
                if high_conf_suggestions:
                    logger.info(
                        "High-confidence tag suggestions found",
                        job_id=job_id,
                        count=len(high_conf_suggestions),
                        suggestions=[{"tag": s["tag_name"], "confidence": round(s["confidence"], 3), "visual": round(s.get("vision_confidence", 0), 3)} for s in high_conf_suggestions[:10]]
                    )
                
                # Get job options from metadata for auto-approve/delete
                job_metadata_query = "SELECT metadata FROM jobs WHERE job_id = $1"
                job_metadata_row = await conn1.fetchrow(job_metadata_query, job_id)
                job_metadata = {}
                if job_metadata_row:
                    # metadata is a JSONB column, so it might be a dict or need parsing
                    metadata_raw = job_metadata_row.get("metadata")
                    if isinstance(metadata_raw, dict):
                        job_metadata = metadata_raw
                    elif metadata_raw:
                        try:
                            import json
                            job_metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                        except:
                            job_metadata = {}
                auto_approve_threshold = job_metadata.get("auto_approve_threshold")
                auto_delete_threshold = job_metadata.get("auto_delete_threshold")
                
                # Store suggestions in database using the same connection
                if suggestions:
                    auto_approved_count = 0
                    auto_deleted_count = 0
                    
                    for suggestion in suggestions:
                        confidence = suggestion['confidence']
                        initial_status = 'pending'
                        
                        # Auto-delete low confidence suggestions if threshold is set
                        if auto_delete_threshold is not None and confidence < auto_delete_threshold:
                            auto_deleted_count += 1
                            continue  # Skip inserting this suggestion
                        
                        # Auto-approve high confidence suggestions if threshold is set
                        if auto_approve_threshold is not None and confidence >= auto_approve_threshold:
                            initial_status = 'auto_applied'
                            auto_approved_count += 1
                        
                        await conn1.execute("""
                            INSERT INTO suggestions (
                                id, job_id, scene_id, tag_id, tag_name, confidence,
                                vision_confidence, asr_confidence, ocr_confidence, 
                                temporal_consistency, calibrated_confidence,
                                evidence_frames, reasoning, signal_details, status, is_backup, created_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, NOW())
                        """,
                        suggestion['id'],
                        suggestion['job_id'],
                        suggestion['scene_id'], 
                        suggestion['tag_id'],
                        suggestion['tag_name'],
                        suggestion['confidence'],
                        suggestion.get('vision_confidence'),
                        suggestion.get('asr_confidence'),
                        suggestion.get('ocr_confidence'),
                        suggestion.get('temporal_consistency'),
                        suggestion.get('calibrated_confidence'),
                        json.dumps(suggestion.get('evidence_frames', [])),
                        suggestion.get('reasoning', ''),
                        json.dumps(suggestion.get('signal_details', {})),
                        initial_status,
                        suggestion.get('is_backup', False)
                        )
                    
                    if auto_approved_count > 0:
                        logger.info("Auto-approved suggestions", job_id=job_id, count=auto_approved_count, threshold=auto_approve_threshold)
                    if auto_deleted_count > 0:
                        logger.info("Auto-deleted low confidence suggestions", job_id=job_id, count=auto_deleted_count, threshold=auto_delete_threshold)
                
                # Calculate metrics from stored suggestions (after auto-delete)
                stored_count_query = "SELECT COUNT(*) FROM suggestions WHERE job_id = $1"
                stored_count = await conn1.fetchval(stored_count_query, job_id) or 0
                
                # Recalculate metrics from stored suggestions
                stored_suggestions_query = "SELECT confidence FROM suggestions WHERE job_id = $1"
                stored_suggestions = await conn1.fetch(stored_suggestions_query, job_id)
                stored_confidences = [s['confidence'] for s in stored_suggestions] if stored_suggestions else []
                
                high_conf = sum(1 for c in stored_confidences if c > 0.8)
                medium_conf = sum(1 for c in stored_confidences if 0.5 < c <= 0.8)
                low_conf = len(stored_confidences) - high_conf - medium_conf
                
                # Get auto-approved/deleted counts (they're in scope from the loop above)
                auto_approved = auto_approved_count if 'auto_approved_count' in locals() else 0
                auto_deleted = auto_deleted_count if 'auto_deleted_count' in locals() else 0
                
                logger.info(
                    "Fusion completed successfully",
                    job_id=job_id,
                    suggestions_generated=stored_count,
                    high_confidence_count=high_conf,
                    medium_confidence_count=medium_conf,
                    low_confidence_count=low_conf,
                    auto_approved=auto_approved,
                    auto_deleted=auto_deleted
                )
                
                return {
                    "success": True,
                    "suggestion_count": stored_count,
                    "high_confidence_count": high_conf,
                    "medium_confidence_count": medium_conf,
                    "low_confidence_count": low_conf,
                    "auto_approved_count": auto_approved,
                    "auto_deleted_count": auto_deleted
                }
                
            finally:
                await conn1.close()
                await conn2.close()
        
        from .database import run_async
        return run_async(run_fusion_pipeline())
        
    except Exception as e:
        logger.error(
            "Fusion processing failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}


async def _update_fusion_progress(conn, job_id: str, progress_data: Dict[str, Any]):
    """Update fusion progress in database"""
    try:
        # Get current progress
        current_progress_query = "SELECT progress FROM jobs WHERE job_id = $1"
        current_progress_row = await conn.fetchrow(current_progress_query, job_id)
        
        if current_progress_row:
            import json
            current_progress = current_progress_row.get('progress') or {}
            if isinstance(current_progress, str):
                current_progress = json.loads(current_progress)
            
            # Update fusion step progress
            current_progress['fusion'] = {**(current_progress.get('fusion', {})), **progress_data}
            
            # Update in database
            update_query = "UPDATE jobs SET progress = $1::jsonb WHERE job_id = $2"
            await conn.execute(update_query, json.dumps(current_progress), job_id)
    except Exception as e:
        logger.warning(f"Failed to update fusion progress: {e}", job_id=job_id)


async def _compute_tag_scores(
    self,
    job_id: str,
    frames: List[Dict[str, Any]],
    tags: List[Dict[str, Any]],
    embedder=None,
    scene_id: str = None,
    conn=None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute multi-modal scores for each tag
    
    Args:
        job_id: Job identifier
        frames: List of frame data with embeddings
        tags: List of active tags with prompts/embeddings
        embedder: CLIP embedder for generating prompt embeddings (optional)
    
    Returns:
        Dict mapping tag_id to score data
    """
    
    tag_scores = {}
    
    # Load CLIP embedder if not provided (needed for multi-prompt matching)
    if embedder is None:
        try:
            from .embeddings import CLIPEmbedder
            from .config import get_worker_settings
            settings = get_worker_settings()
            model_name = getattr(settings, 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            vision_device = getattr(settings, 'vision_device', 'auto')
            device = None if vision_device == 'auto' else vision_device
            embedder = CLIPEmbedder(model_name=model_name, device=device)
            logger.info("Loaded CLIP embedder for multi-prompt matching", job_id=job_id)
        except Exception as e:
            logger.warning(
                "Failed to load CLIP embedder, using stored embeddings only",
                job_id=job_id,
                error=str(e)
            )
            embedder = None
    
    total_tags = len(tags)
    for tag_idx, tag in enumerate(tags, 1):
        tag_id = tag["tag_id"]
        tag_name = tag["name"]
        prompts = tag.get("prompts", [tag_name])
        tag_embedding = np.array(tag.get("embedding", []))
        
        # Log progress for each tag
        progress_info = {
            "tag_prog": f"{tag_idx}/{total_tags}",
            "tag_name": tag_name,
            "job_id": job_id
        }
        if scene_id:
            progress_info["scene_id"] = scene_id
        
        # Update fusion progress in database every 10 tags or on first/last tag
        if conn and (tag_idx % 10 == 0 or tag_idx == 1 or tag_idx == total_tags):
            await _update_fusion_progress(conn, job_id, {
                "status": "processing",
                "current_step": "Calculating tag scores",
                "tags_total": total_tags,
                "tags_analyzed": tag_idx,
                "current_tag": tag_name,
                "tag_progress": f"{tag_idx}/{total_tags}"
            })
        
        # Detect generic template prompts
        generic_patterns = [
            "scene featuring", "visual content showing", "frame displaying", 
            "image containing", "visual representation of"
        ]
        has_generic_prompts = any(
            any(pattern in prompt.lower() for pattern in generic_patterns)
            for prompt in prompts
        )
        all_generic = all(
            any(pattern in prompt.lower() for pattern in generic_patterns)
            for prompt in prompts
        )
        
        if all_generic:
            logger.warning(
                "Tag has only generic template prompts",
                job_id=job_id,
                tag_id=tag_id,
                tag_name=tag_name,
                prompts=prompts,
                message="Consider improving prompts with visual descriptions. Generic prompts like 'Scene featuring X' don't help CLIP match visual features."
            )
        elif has_generic_prompts:
            logger.debug(
                "Tag has some generic prompts",
                job_id=job_id,
                tag_id=tag_id,
                tag_name=tag_name,
                prompt_count=len(prompts)
            )
        
        # Compute visual similarity scores using ALL prompts
        # This is the key fix: compute similarity against all prompts and take max
        visual_diagnostics = None
        if embedder and prompts:
            visual_scores, visual_diagnostics = _compute_visual_scores_multi_prompt(self, frames, prompts, embedder, tag_embedding)
        elif tag_embedding.size > 0:
            # Fallback to single stored embedding if no embedder available
            visual_scores = _compute_visual_scores(self, frames, tag_embedding)
            # For fallback, create minimal diagnostics
            raw_scores = visual_scores  # In fallback mode, normalized scores are the raw scores
            visual_diagnostics = {
                "raw_scores": raw_scores,
                "prompt_similarities": {},
                "max_prompt_indices": [],
                "prompts_used": ["stored_embedding"]
            }
        else:
            logger.warning(
                "Tag missing embedding and no embedder available",
                job_id=job_id,
                tag_id=tag_id,
                tag_name=tag_name
            )
            continue
        
        # Extract raw CLIP scores from diagnostics
        raw_clip_scores = visual_diagnostics.get("raw_scores", []) if visual_diagnostics else []
        
        # Compute text match scores
        text_scores = _compute_text_scores(self, frames, tag_name)
        
        # Combine scores with weighted fusion (pass tag_name for adaptive scoring)
        combined_score = _fuse_scores(self, visual_scores, text_scores, tag_name=tag_name)
        
        # Apply temporal consistency (pass tag_name for adaptive handling of high-precision tags)
        temporal_score = _apply_temporal_consistency(self, visual_scores, tag_name=tag_name)
        
        # Final aggregated score (pass tag_name for adaptive handling of high-precision tags)
        final_score = _aggregate_scores(
            self,
            combined_score,
            temporal_score,
            visual_scores,
            text_scores,
            tag_name=tag_name
        )
        
        # Find evidence frames (top matching frames)
        evidence_frames = _find_evidence_frames(self, frames, visual_scores, text_scores)
        
        # Calculate statistics for raw CLIP scores
        raw_score_stats = {}
        if raw_clip_scores:
            raw_score_stats = {
                "min": float(np.min(raw_clip_scores)),
                "max": float(np.max(raw_clip_scores)),
                "mean": float(np.mean(raw_clip_scores)),
                "median": float(np.median(raw_clip_scores)),
                "std": float(np.std(raw_clip_scores))
            }
        
        # Calculate statistics for normalized visual scores
        normalized_visual_mean = np.mean(visual_scores) if visual_scores else 0.0
        normalized_visual_stats = {
            "min": float(np.min(visual_scores)) if visual_scores else 0.0,
            "max": float(np.max(visual_scores)) if visual_scores else 0.0,
            "mean": float(normalized_visual_mean),
            "median": float(np.median(visual_scores)) if visual_scores else 0.0
        }
        
        tag_scores[tag_id] = {
            "tag_id": tag_id,
            "tag_name": tag_name,
            "score": final_score,
            "visual_score": normalized_visual_mean,
            "text_score": np.mean(text_scores) if text_scores else 0.0,
            "temporal_score": temporal_score,
            "evidence_frames": evidence_frames,
            "match_count": len([s for s in visual_scores if s > 0.6]),
            "raw_clip_stats": raw_score_stats,
            "normalized_visual_stats": normalized_visual_stats,
            "combined_score": combined_score,
            "prompts_used": visual_diagnostics.get("prompts_used", []) if visual_diagnostics else prompts
        }
        
        # Log detailed score pipeline diagnostics with progress info
        log_data = {
            **progress_info,
            "final_score": round(final_score, 4),
            "raw_clip_min": round(raw_score_stats.get("min", 0), 4) if raw_score_stats else None,
            "raw_clip_max": round(raw_score_stats.get("max", 0), 4) if raw_score_stats else None,
            "raw_clip_mean": round(raw_score_stats.get("mean", 0), 4) if raw_score_stats else None,
            "raw_clip_median": round(raw_score_stats.get("median", 0), 4) if raw_score_stats else None,
            "normalized_visual_mean": round(normalized_visual_mean, 4),
            "combined_score": round(combined_score, 4),
            "temporal_score": round(temporal_score, 4),
            "text_score": round(np.mean(text_scores) if text_scores else 0.0, 4),
            "prompt_count": len(prompts),
            "has_generic_prompts": has_generic_prompts
        }
        logger.info("Tag scores computed with diagnostics", **log_data)
        
        # Log which prompt was most effective (if multi-prompt) with progress info
        if visual_diagnostics and len(visual_diagnostics.get("prompts_used", [])) > 1:
            prompt_sims = visual_diagnostics.get("prompt_similarities", {})
            if prompt_sims:
                prompt_avg_sims = {
                    prompt: np.mean(sims) if sims else 0.0
                    for prompt, sims in prompt_sims.items()
                }
                best_prompt = max(prompt_avg_sims.items(), key=lambda x: x[1])
                logger.debug(
                    "Prompt effectiveness analysis",
                    **progress_info,
                    best_prompt=best_prompt[0][:50] + "..." if len(best_prompt[0]) > 50 else best_prompt[0],
                    best_prompt_avg_similarity=round(best_prompt[1], 4),
                    all_prompt_avgs={k[:30]: round(v, 4) for k, v in prompt_avg_sims.items()}
                )
    
    return tag_scores


def _compute_visual_scores_multi_prompt(
    self,
    frames: List[Dict[str, Any]],
    prompts: List[str],
    embedder,
    fallback_embedding: np.ndarray = None
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Compute visual similarity scores between frames and tag using ALL prompts.
    Takes the maximum similarity across all prompts for each frame.
    
    This is the correct approach: compute similarity against all prompts and take max,
    rather than using a single stored embedding.
    
    Args:
        frames: List of frames with embeddings
        prompts: List of prompt strings for the tag
        embedder: CLIP embedder for generating prompt embeddings
        fallback_embedding: Optional fallback embedding if prompts fail
    
    Returns:
        Tuple of:
        - List of normalized similarity scores (one per frame) - max similarity across all prompts
        - Dict with diagnostic info: raw_scores, prompt_similarities, max_prompt_indices
    """
    
    if not prompts:
        # No prompts, use fallback or return zeros
        if fallback_embedding is not None and fallback_embedding.size > 0:
            scores = _compute_visual_scores(self, frames, fallback_embedding)
            return scores, {"raw_scores": scores, "prompt_similarities": {}, "max_prompt_indices": []}
        return [0.0] * len(frames), {"raw_scores": [0.0] * len(frames), "prompt_similarities": {}, "max_prompt_indices": []}
    
    # Generate embeddings for all prompts (batch for efficiency)
    prompt_embeddings = []
    try:
        # Try to encode all prompts at once if embedder supports it
        # Otherwise encode one by one
        if hasattr(embedder, 'encode_text_batch'):
            try:
                prompt_embeddings = embedder.encode_text_batch(prompts)
            except:
                # Fallback to individual encoding
                for prompt in prompts:
                    try:
                        prompt_emb = embedder.encode_text(prompt)
                        prompt_embeddings.append(prompt_emb)
                    except Exception as e:
                        logger.debug(f"Failed to encode prompt: {prompt[:50]}... Error: {e}")
                        continue
        else:
            # Encode prompts individually
            for prompt in prompts:
                try:
                    prompt_emb = embedder.encode_text(prompt)
                    prompt_embeddings.append(prompt_emb)
                except Exception as e:
                    logger.debug(f"Failed to encode prompt: {prompt[:50]}... Error: {e}")
                    continue
    except Exception as e:
        logger.warning(f"Failed to generate prompt embeddings: {e}")
        prompt_embeddings = []
    
    if not prompt_embeddings:
        # All prompts failed, use fallback
        if fallback_embedding is not None and fallback_embedding.size > 0:
            logger.debug("Using fallback embedding for tag")
            return _compute_visual_scores(self, frames, fallback_embedding)
        return [0.0] * len(frames)
    
    # Normalize all prompt embeddings once (more efficient)
    prompt_embeddings_norm = []
    for prompt_emb in prompt_embeddings:
        prompt_emb_np = np.array(prompt_emb)
        prompt_emb_np = prompt_emb_np / (np.linalg.norm(prompt_emb_np) + 1e-8)
        prompt_embeddings_norm.append(prompt_emb_np)
    
    # For each frame, compute similarity against all prompts and take max
    scores = []
    raw_scores = []
    prompt_similarities = {i: [] for i in range(len(prompts))}  # Track similarity per prompt
    max_prompt_indices = []  # Track which prompt had max similarity for each frame
    
    for frame in frames:
        frame_embedding = frame.get("embedding")
        
        if not frame_embedding:
            scores.append(0.0)
            raw_scores.append(0.0)
            max_prompt_indices.append(-1)
            for i in range(len(prompts)):
                prompt_similarities[i].append(0.0)
            continue
        
        frame_emb = np.array(frame_embedding)
        
        # Validate embedding dimensions match
        frame_dim = len(frame_emb)
        if prompt_embeddings_norm:
            prompt_dim = len(prompt_embeddings_norm[0])
            if frame_dim != prompt_dim:
                logger.warning(
                    f"Embedding dimension mismatch: frame={frame_dim}, prompt={prompt_dim}. "
                    f"This may indicate model mismatch. Skipping frame.",
                    job_id=getattr(self, 'job_id', 'unknown')
                )
                scores.append(0.0)
                raw_scores.append(0.0)
                max_prompt_indices.append(-1)
                for i in range(len(prompts)):
                    prompt_similarities[i].append(0.0)
                continue
        
        # Normalize frame embedding
        frame_emb = frame_emb / (np.linalg.norm(frame_emb) + 1e-8)
        
        # Compute similarity against all prompts
        frame_similarities = []
        for prompt_idx, prompt_emb_norm in enumerate(prompt_embeddings_norm):
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(frame_emb, prompt_emb_norm))
            prompt_similarities[prompt_idx].append(similarity)
            frame_similarities.append(similarity)
        
        # Apply pooling method: MAX or Softmax
        pooling_method = self.settings.prompt_pooling_method if hasattr(self.settings, 'prompt_pooling_method') else "max"
        tau = self.settings.prompt_pooling_temperature if hasattr(self.settings, 'prompt_pooling_temperature') else 0.07
        
        if pooling_method == "softmax" and len(frame_similarities) > 0:
            # Softmax pooling: τ * log(mean(exp(sim_i/τ)))
            # This smooths out spiky MAX scores and considers all prompts
            similarities_array = np.array(frame_similarities)
            # Clip values to prevent overflow in exp
            similarities_array = np.clip(similarities_array, -10, 10)
            softmax_score = tau * np.log(np.mean(np.exp(similarities_array / tau)) + 1e-10)
            max_similarity = float(softmax_score)
            # Find which prompt contributed most (for diagnostics)
            max_prompt_idx = int(np.argmax(frame_similarities))
        else:
            # Original MAX pooling (default for backward compatibility)
            max_similarity = max(frame_similarities) if frame_similarities else -1.0
            max_prompt_idx = int(np.argmax(frame_similarities)) if frame_similarities else -1
        
        # Store raw CLIP score before normalization
        raw_scores.append(max_similarity)
        max_prompt_indices.append(max_prompt_idx)
        
        # Apply normalization if enabled (configurable for testing)
        enable_normalization = self.settings.enable_score_normalization if hasattr(self.settings, 'enable_score_normalization') else True
        
        # Detect model type for model-specific scaling
        # Get model_name from the embedder (which is passed to this function)
        is_siglip = False
        try:
            if embedder and hasattr(embedder, 'model_name'):
                model_name = embedder.model_name
                is_siglip = "siglip" in model_name.lower()
            elif hasattr(embedder, 'is_siglip'):
                # CLIPEmbedder has is_siglip attribute
                is_siglip = embedder.is_siglip
            else:
                # Fallback: check settings
                model_name = getattr(self.settings, 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                is_siglip = "siglip" in model_name.lower()
        except:
            # Default to CLIP if we can't determine
            is_siglip = False
        
        if enable_normalization:
            # Model-specific scaling: SigLIP produces lower similarity scores than CLIP
            # Based on observations: SigLIP scores are ~0.03-0.11 for good matches
            # CLIP scores are ~0.20-0.50 for similar content
            # We need to scale SigLIP scores to match CLIP's scale for consistent thresholds
            
            if is_siglip:
                # SigLIP model: Scale scores to match CLIP's distribution
                # Empirical observation: SigLIP scores are ~0.03-0.11 for good matches
                # CLIP scores are ~0.20-0.50 for similar content
                # Using 2.5x scaling to bring SigLIP scores into CLIP's range
                # This ensures consistent thresholds across models
                raw_siglip_score = max_similarity
                scaled_similarity = max_similarity * 2.5
                
                # Log scaling for debugging (only for first frame to avoid spam)
                if len(raw_scores) == 0:  # First frame
                    logger.debug(
                        "SigLIP score scaling applied",
                        raw_score=round(raw_siglip_score, 4),
                        scaled_score=round(scaled_similarity, 4),
                        scaling_factor=2.5
                    )
                
                # Apply CLIP-style normalization to scaled scores
                if scaled_similarity > 0.75:
                    normalized = 0.78 + (scaled_similarity - 0.75) * 0.88
                elif scaled_similarity > 0.60:
                    normalized = 0.62 + (scaled_similarity - 0.60) * 1.07
                elif scaled_similarity > 0.45:
                    normalized = 0.46 + (scaled_similarity - 0.45) * 1.07
                elif scaled_similarity > 0.25:
                    normalized = scaled_similarity * 1.05
                elif scaled_similarity > 0.20:
                    normalized = scaled_similarity
                elif scaled_similarity > 0.18:
                    normalized = scaled_similarity
                elif scaled_similarity > 0.15:
                    normalized = scaled_similarity * 0.95
                elif scaled_similarity > 0.13:
                    normalized = scaled_similarity
                else:
                    normalized = max(0.0, scaled_similarity * 0.9)
            else:
                # CLIP model: Use existing normalization
                # CLIP typically gives:
                # - 0.20-0.30: unrelated content
                # - 0.30-0.45: weak match / generic similarity
                # - 0.45-0.60: moderate match
                # - 0.60-0.75: good match
                # - 0.75-0.85: strong match
                # - 0.85+: very strong match
                
                # CRITICAL FIX: Preserve and amplify CLIP scores in the 0.20-0.30 range
                # Diagnostic data shows: valid matches are 0.17-0.22, invalid are 0.13-0.14
                # We need to PRESERVE this separation, not compress it away!
                if max_similarity > 0.75:
                    # Very strong match: minimal boost
                    normalized = 0.78 + (max_similarity - 0.75) * 0.88  # Maps 0.75->0.78, 1.0->1.0
                elif max_similarity > 0.60:
                    # Good match: slight boost
                    normalized = 0.62 + (max_similarity - 0.60) * 1.07  # Maps 0.60->0.62, 0.75->0.78
                elif max_similarity > 0.45:
                    # Moderate match: minimal adjustment
                    normalized = 0.46 + (max_similarity - 0.45) * 1.07  # Maps 0.45->0.46, 0.60->0.62
                elif max_similarity > 0.25:
                    # Good match: preserve with slight amplification
                    normalized = max_similarity * 1.05  # Slight amplification
                elif max_similarity > 0.20:
                    # Moderate match: preserve (CRITICAL FIX for brief tags)
                    # Brief tags near threshold benefit from score preservation in this range
                    normalized = max_similarity  # No compression, no amplification
                elif max_similarity > 0.18:
                    # Weak-to-moderate match: preserve (helps brief tags near threshold)
                    normalized = max_similarity  # No compression
                elif max_similarity > 0.15:
                    # Weak match: slight compression to separate from valid matches
                    normalized = max_similarity * 0.95  # Minimal compression
                elif max_similarity > 0.13:
                    # Very weak match: preserve to avoid over-compression
                    normalized = max_similarity  # No compression
                else:
                    # Very weak match: compress to separate from valid matches
                    normalized = max(0.0, max_similarity * 0.9)  # Compress noise
            
            normalized = max(0.0, min(1.0, normalized))
        else:
            # No normalization - use raw scores directly (may need scaling for SigLIP)
            if is_siglip:
                # Even without normalization, scale SigLIP scores for consistency
                normalized = max_similarity * 2.5
                normalized = max(0.0, min(1.0, normalized))
            else:
                normalized = max_similarity
        
        scores.append(normalized)
    
    # Build diagnostic info
    diagnostics = {
        "raw_scores": raw_scores,
        "prompt_similarities": {prompts[i]: prompt_similarities[i] for i in range(len(prompts))},
        "max_prompt_indices": max_prompt_indices,
        "prompts_used": prompts
    }
    
    return scores, diagnostics


def _compute_visual_scores(
    self,
    frames: List[Dict[str, Any]],
    tag_embedding: np.ndarray
) -> List[float]:
    """
    Compute visual similarity scores between frames and tag (single embedding)
    
    Args:
        frames: List of frames with embeddings
        tag_embedding: Tag text embedding
    
    Returns:
        List of similarity scores (one per frame)
    """
    
    scores = []
    
    for frame in frames:
        frame_embedding = frame.get("embedding")
        
        if not frame_embedding:
            scores.append(0.0)
            continue
        
        frame_emb = np.array(frame_embedding)
        tag_emb = np.array(tag_embedding)
        
        # Ensure embeddings are normalized
        frame_emb = frame_emb / (np.linalg.norm(frame_emb) + 1e-8)
        tag_emb = tag_emb / (np.linalg.norm(tag_emb) + 1e-8)
        
        # Compute cosine similarity (dot product of normalized vectors)
        # Range is [-1, 1], but CLIP embeddings typically produce [0.2, 0.9] for similar content
        similarity = float(np.dot(frame_emb, tag_emb))
        
        # Apply normalization if enabled (configurable for testing)
        enable_normalization = self.settings.enable_score_normalization if hasattr(self.settings, 'enable_score_normalization') else True
        if enable_normalization:
            # CRITICAL FIX: Preserve and amplify CLIP scores in the 0.20-0.30 range
            # Diagnostic data shows: valid matches are 0.17-0.22, invalid are 0.13-0.14
            if similarity > 0.75:
                # Very strong match: minimal boost
                normalized = 0.78 + (similarity - 0.75) * 0.88  # Maps 0.75->0.78, 1.0->1.0
            elif similarity > 0.60:
                # Good match: slight boost
                normalized = 0.62 + (similarity - 0.60) * 1.07  # Maps 0.60->0.62, 0.75->0.78
            elif similarity > 0.45:
                # Moderate match: minimal adjustment
                normalized = 0.46 + (similarity - 0.45) * 1.07  # Maps 0.45->0.46, 0.60->0.62
            elif similarity > 0.20:
                # Moderate match: preserve with minimal amplification
                normalized = similarity * 1.1  # Minimal amplification for moderate matches
            elif similarity > 0.16:
                # Weak match: preserve mostly (don't amplify too much)
                normalized = similarity * 1.05  # Very slight amplification
            elif similarity > 0.13:
                # Very weak match: preserve or slight compression
                normalized = similarity  # No amplification, no compression
            else:
                # Very weak match: compress to separate from valid matches
                normalized = max(0.0, similarity * 0.9)  # Compress noise more
            
            normalized = max(0.0, min(1.0, normalized))
        else:
            # No normalization - use raw CLIP scores directly
            normalized = similarity
        
        scores.append(normalized)
    
    return scores


def _compute_text_scores(
    self,
    frames: List[Dict[str, Any]],
    tag_name: str
) -> List[float]:
    """
    Compute text match scores (ASR/OCR mentions of tag)
    
    Args:
        frames: List of frames with text analysis
        tag_name: Tag name to search for
    
    Returns:
        List of match scores (one per frame)
    """
    
    scores = []
    tag_keywords = _extract_keywords(self, tag_name)
    
    for frame in frames:
        # Check if frame has associated text (OCR or nearby ASR)
        text_content = frame.get("text_content", "")
        
        if not text_content:
            scores.append(0.0)
            continue
        
        # Compute keyword match score
        text_lower = text_content.lower()
        match_score = 0.0
        
        for keyword in tag_keywords:
            if keyword.lower() in text_lower:
                # Exact match - high score
                match_score = max(match_score, 0.9)
            elif _fuzzy_match(self, keyword, text_content):
                # Fuzzy match - medium score
                match_score = max(match_score, 0.6)
        
        scores.append(match_score)
    
    return scores


def _extract_keywords(self, tag_name: str) -> List[str]:
    """
    Extract keywords from tag name
    
    Args:
        tag_name: Tag name
    
    Returns:
        List of keywords
    """
    
    # Split on common separators
    keywords = []
    
    # Handle multi-word tags
    for word in tag_name.replace("-", " ").replace("_", " ").split():
        if len(word) > 2:  # Skip very short words
            keywords.append(word)
    
    # Also include full tag name
    keywords.append(tag_name)
    
    return keywords


def _fuzzy_match(self, keyword: str, text: str, threshold: float = 0.8) -> bool:
    """
    Check if keyword fuzzy matches text
    
    Args:
        keyword: Keyword to match
        text: Text to search in
        threshold: Similarity threshold (0.0 to 1.0)
    
    Returns:
        True if fuzzy match found
    """
    
    from difflib import SequenceMatcher
    
    keyword_lower = keyword.lower()
    text_lower = text.lower()
    
    # Check all substrings of similar length
    for i in range(len(text_lower) - len(keyword_lower) + 1):
        substring = text_lower[i:i + len(keyword_lower)]
        similarity = SequenceMatcher(None, keyword_lower, substring).ratio()
        
        if similarity >= threshold:
            return True
    
    return False


def _get_tag_classification_for_scoring(tag_name: str) -> str:
    """Get tag classification for adaptive scoring (brief, common, or high_precision).

    Classifications are loaded from the TAG_SCORING_CLASSIFICATIONS env-var (JSON)
    or default to 'common' for every tag.

    Example env value:
        TAG_SCORING_CLASSIFICATIONS={"brief":["Forest","Tattoos"],"high_precision":["Outdoor","Night"]}
    """
    import json as _json

    raw = os.environ.get("TAG_SCORING_CLASSIFICATIONS", "")
    if raw:
        try:
            classifications = _json.loads(raw)
        except Exception:
            classifications = {}
    else:
        classifications = {}

    tag_lower = tag_name.lower()
    for brief_tag in classifications.get("brief", []):
        if brief_tag.lower() in tag_lower:
            return "brief"
    for hp_tag in classifications.get("high_precision", []):
        if hp_tag.lower() in tag_lower:
            return "high_precision"
    return "common"

def _fuse_scores(
    self,
    visual_scores: List[float],
    text_scores: List[float],
    tag_name: str = None
) -> float:
    """
    Fuse visual and text scores with weighted combination
    
    Args:
        visual_scores: List of visual similarity scores
        text_scores: List of text match scores
        tag_name: Tag name for adaptive scoring (optional)
    
    Returns:
        Combined score (0.0 to 1.0)
    """
    
    # Determine scoring method: adaptive based on tag classification if tag_name provided
    if tag_name:
        tag_classification = _get_tag_classification_for_scoring(tag_name)
        if tag_classification == "brief":
            # Brief tags: use "max" scoring (best for sparse appearances)
            scoring_method = "max"
        elif tag_classification == "high_precision":
            # High-precision tags: use "mean" scoring (more conservative, reduces false positives)
            scoring_method = "mean"
        else:
            # Common tags: use default from settings
            scoring_method = self.settings.scoring_method if hasattr(self.settings, 'scoring_method') else "mean"
    else:
        # No tag_name provided: use default from settings
        scoring_method = self.settings.scoring_method if hasattr(self.settings, 'scoring_method') else "mean"
    
    # Compute visual score based on scoring method
    if scoring_method == "max_frequency":
        # Max*frequency scoring: rewards both accuracy (max similarity) and consistency (times seen)
        # IMPROVED: Use adaptive formula that works better for brief tags
        if not visual_scores:
            visual_score = 0.0
        else:
            max_similarity = max(visual_scores)
            # Use a lower threshold for frequency calculation to catch brief tags
            enable_normalization = self.settings.enable_score_normalization if hasattr(self.settings, 'enable_score_normalization') else True
            if enable_normalization:
                frequency_threshold = 0.18  # Lower threshold to catch brief tags (was 0.20)
            else:
                frequency_threshold = 0.15  # Lower threshold for raw CLIP scores
            
            frames_above_threshold = sum(1 for s in visual_scores if s > frequency_threshold)
            frequency = frames_above_threshold / len(visual_scores) if visual_scores else 0.0
            
            # IMPROVED FORMULA: Use adaptive weighting based on frequency
            # For very brief tags (frequency < 5%), use mostly max_similarity (frequency is unreliable)
            # For brief tags (5-20% frequency), use balanced weighting
            # For common tags (>20% frequency), use frequency weighting
            if frequency < 0.05:
                # Very brief tag (<5% of frames): use 70% of max_similarity (frequency is too unreliable)
                visual_score = max_similarity * 0.70
            elif frequency < 0.20:
                # Brief tag (5-20% of frames): use balanced weighting
                # Formula: score = max_similarity * (0.5 + 0.5 * frequency)
                # Example: freq=0.10: 0.1953 * (0.5 + 0.5*0.10) = 0.1953 * 0.55 = 0.107
                visual_score = max_similarity * (0.5 + 0.5 * frequency)
            else:
                # Common tag (>=20% of frames): use standard frequency weighting
                visual_score = max_similarity * frequency
    elif scoring_method == "max":
        # Max scoring: use maximum similarity (best for brief tags)
        visual_score = max(visual_scores) if visual_scores else 0.0
    else:
        # Default: mean scoring
        visual_score = np.mean(visual_scores) if visual_scores else 0.0
    
    text_avg = np.mean(text_scores) if text_scores else 0.0
    
    # If text scores are all zero (ASR/OCR not used), just return visual score
    # Otherwise, use weighted combination
    if text_avg == 0.0 or not any(text_scores):
        # ASR/OCR not used - visual-only scoring
        combined = visual_score
    else:
        # ASR/OCR used - weighted combination
        visual_weight = self.settings.vision_weight if hasattr(self.settings, 'vision_weight') else 0.7
        text_weight = 1.0 - visual_weight
        combined = (visual_score * visual_weight) + (text_avg * text_weight)
    
    return float(combined)


def _apply_temporal_consistency(self, visual_scores: List[float], tag_name: str = None) -> float:
    """
    Apply temporal consistency - repeated signals across frames
    
    Args:
        visual_scores: List of visual scores across frames
        tag_name: Tag name for adaptive handling (optional)
    
    Returns:
        Temporal consistency score (0.0 to 1.0)
    """
    
    if not visual_scores:
        return 0.0
    
    # IMPROVED: Use adaptive threshold based on tag classification
    # High-precision tags need higher threshold to avoid false positives
    scoring_method = self.settings.scoring_method if hasattr(self.settings, 'scoring_method') else "mean"
    enable_normalization = self.settings.enable_score_normalization if hasattr(self.settings, 'enable_score_normalization') else True
    
    # Get tag classification for adaptive threshold
    if tag_name:
        tag_classification = _get_tag_classification_for_scoring(tag_name)
    else:
        tag_classification = "common"
    
    # IMPROVED: Use adaptive threshold based on tag classification
    # Problem: False positives get temporal_score=0.99 with threshold=0.18
    # This is because they have many frames with scores just above 0.18, giving high consistency
    # Solution: Use higher threshold for high-precision tags to avoid false positives
    if tag_classification == "high_precision":
        # High-precision tags: use higher threshold (0.22-0.25) to reduce false positives
        # These tags are prone to false positives, so we need stricter temporal consistency
        threshold = 0.23 if enable_normalization else 0.22
    elif scoring_method in ["max", "max_frequency"]:
        # Brief tags: use moderate threshold (not too low to avoid false positives)
        # But still lower than common tags to catch brief appearances
        threshold = 0.20 if enable_normalization else 0.18
    else:
        # Common tags: use standard threshold
        threshold = self.settings.temporal_consistency_threshold if hasattr(self.settings, 'temporal_consistency_threshold') else 0.25
    
    # Count frames with scores above threshold
    high_score_count = sum(1 for s in visual_scores if s > threshold)
    consistency = high_score_count / len(visual_scores) if visual_scores else 0.0
    
    # IMPROVED: Better handling of brief tags while avoiding false positives
    # Problem: False positives were getting consistency=0.99 with low threshold
    # Solution: Cap temporal consistency boost and use non-linear scaling, especially for high-precision tags
    if tag_classification == "high_precision":
        # High-precision tags: aggressively cap temporal consistency to prevent false positives
        # Solution: Cap at lower values and reduce scaling
        if consistency > 0.6:
            # Very high consistency: cap at 0.4 to prevent false positives
            scaled_consistency = 0.4
        elif consistency > 0.4:
            # High consistency: scale down more aggressively
            scaled_consistency = 0.3 + (consistency - 0.4) * 0.5  # Maps 0.4->0.3, 0.6->0.4
        elif consistency > 0.2:
            # Moderate consistency: scale down
            scaled_consistency = 0.15 + (consistency - 0.2) * 0.75  # Maps 0.2->0.15, 0.4->0.3
        else:
            # Low consistency: minimal weight
            scaled_consistency = consistency * 0.5
    elif scoring_method in ["max", "max_frequency"]:
        # Brief tags: scale more generously but cap to avoid false positives
        if consistency > 0.7:
            # Very high consistency (70%+): cap at 0.7 to avoid false positives
            scaled_consistency = 0.7
        elif consistency > 0.5:
            # High consistency (50-70%): scale linearly
            scaled_consistency = consistency
        elif consistency > 0.2:
            # Moderate consistency (20-50%): scale more generously for brief tags
            scaled_consistency = 0.2 + (consistency - 0.2) * 1.0  # Linear scaling
        elif consistency > 0.05:
            # Low consistency (5-20%): still meaningful for brief tags
            scaled_consistency = consistency * 1.2  # Moderate boost (reduced from 1.5)
        else:
            # Very low consistency (<5%): minimal weight
            scaled_consistency = consistency * 0.5
    else:
        # Common tags: use standard scaling
        if consistency > 0.5:
            scaled_consistency = consistency
        elif consistency > 0.3:
            scaled_consistency = 0.3 + (consistency - 0.3) * 0.7
        else:
            scaled_consistency = consistency * 0.5
    
    return float(scaled_consistency)


def _aggregate_scores(
    self,
    combined_score: float,
    temporal_score: float,
    visual_scores: List[float],
    text_scores: List[float],
    tag_name: str = None
) -> float:
    """
    Aggregate all scores into final calibrated confidence
    
    Args:
        combined_score: Weighted fusion score
        temporal_score: Temporal consistency score
        visual_scores: Individual visual scores
        text_scores: Individual text scores
        tag_name: Tag name for adaptive handling (optional)
    
    Returns:
        Final calibrated score (0.0 to 1.0)
    """
    
    # Base score from fusion
    base_score = combined_score
    
    # Get tag classification for adaptive handling
    if tag_name:
        tag_classification = _get_tag_classification_for_scoring(tag_name)
    else:
        tag_classification = "common"
    
    # Check if aggregation boosts are enabled (configurable for testing)
    enable_boosts = self.settings.enable_aggregation_boosts if hasattr(self.settings, 'enable_aggregation_boosts') else True
    
    if enable_boosts:
        # CRITICAL FIX: Dramatically reduce temporal boost for high-precision tags to prevent false positives
        # Problem: High-precision false positives get boosted by temporal_score
        # Solution: Use much smaller temporal boost for high-precision tags, or disable it entirely
        if tag_classification == "high_precision":
            # High-precision tags: minimal or no temporal boost to prevent false positives
            # These tags are prone to false positives, so temporal consistency should be very conservative
            if base_score > 0.22 and temporal_score > 0.6:
                # Only apply tiny boost for very high base score AND very high temporal
                temporal_boost = temporal_score * 0.02 * base_score  # Max: 0.4 * 0.02 * 0.25 = 0.002 (0.2%)
            else:
                # No boost for high-precision tags with moderate scores (prevents false positives)
                temporal_boost = 0.0
        elif base_score > 0.18 and temporal_score > 0.5:
            # Only apply temporal boost if base score is decent AND temporal consistency is high
            # This prevents false positives from getting boosted
            # Scale boost based on base score: max 5% boost for very high base + high temporal
            temporal_boost = temporal_score * 0.05 * base_score  # Max: 1.0 * 0.05 * 0.25 = 0.0125 (1.25%)
        elif base_score > 0.15 and temporal_score > 0.7:
            # Moderate base score with high temporal: small boost
            temporal_boost = temporal_score * 0.03 * base_score  # Max: 1.0 * 0.03 * 0.20 = 0.006 (0.6%)
        else:
            # Low base score or low temporal: no boost (likely false positive)
            temporal_boost = 0.0
        
        # Boost from strong individual signals (peak signals are important)
        max_visual = max(visual_scores) if visual_scores else 0.0
        max_text = max(text_scores) if text_scores else 0.0
        mean_visual = np.mean(visual_scores) if visual_scores else 0.0
        
        # CRITICAL FIX: Extend peak boosts to moderate scores (0.20-0.60 range)
        # Based on diagnostic data, valid matches have max_visual ~0.20-0.25, but get no boost
        if max_visual > 0.75 or max_text > 0.75:
            # Very strong signal: give moderate boost
            peak_boost = (max(max_visual, max_text) - 0.75) * 0.3  # Up to ~7.5% boost for 1.0 score
        elif max_visual > 0.6 or max_text > 0.6:
            # Good signal: small boost
            peak_boost = (max(max_visual, max_text) - 0.6) * 0.2  # Up to ~3% boost for 0.75 score
        elif max_visual > 0.35 or max_text > 0.35:
            # Moderate signal: small boost (NEW - covers 0.35-0.60 range)
            peak_boost = (max(max_visual, max_text) - 0.35) * 0.12  # Up to ~3% boost for 0.60 score
        elif max_visual > 0.22 or max_text > 0.22:
            # Moderate signal: small boost (covers 0.22-0.35 range)
            peak_boost = (max(max_visual, max_text) - 0.22) * 0.10  # Up to ~1.3% boost for 0.35 score
        else:
            # Low scores (< 0.22) won't get boost, which is correct
            peak_boost = 0.0
        
        # REMOVED: Consistency boost was causing false positives
        # The consistency boost was adding to tags with high mean_visual but low discriminative power
        # Solution: Remove consistency boost - rely on base score and peak signals instead
        consistency_boost = 0.0
        
        # Multimodal boost (ASR/OCR not used, so this will rarely trigger)
        # Keep it for future use, but since text_score is always 0.0, this won't apply
        if max_visual > 0.65 and max_text > 0.5:
            multimodal_boost = 0.05  # 5% bonus for multimodal agreement
        else:
            multimodal_boost = 0.0
        
        # Final score with all boosts
        final_score = base_score + temporal_boost + peak_boost + consistency_boost + multimodal_boost
    else:
        # No aggregation boosts - use combined_score directly as final_score
        final_score = base_score
    
    # REMOVED: Non-linear scaling was causing issues
    # Scaling was making false positives score even higher
    # Solution: Keep final score as-is (no scaling) to preserve accuracy
    # The boosts above should be sufficient to separate valid from invalid matches
    
    # Clamp to [0, 1]
    final_score = max(0.0, min(1.0, final_score))
    
    return float(final_score)


def _find_evidence_frames(
    self,
    frames: List[Dict[str, Any]],
    visual_scores: List[float],
    text_scores: List[float],
    top_k: int = 3
) -> List[str]:
    """
    Find top evidence frames for tag suggestion
    
    Args:
        frames: List of frames
        visual_scores: Visual similarity scores
        text_scores: Text match scores
        top_k: Number of evidence frames to return
    
    Returns:
        List of frame IDs with strongest evidence
    """
    
    # Combine scores for ranking
    combined_scores = [
        (v * 0.7 + t * 0.3) for v, t in zip(visual_scores, text_scores)
    ]
    
    # Get top-k frames
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    
    evidence_frame_ids = [frames[i]["id"] for i in top_indices if i < len(frames)]
    
    return evidence_frame_ids


def _generate_tag_suggestions(
    self,
    job_id: str,
    tag_scores: Dict[str, Dict[str, Any]],
    frames: List[Dict[str, Any]],
    tag_hierarchy: Dict[str, Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate tag suggestions from computed scores with hierarchical filtering
    
    Args:
        job_id: Job identifier
        tag_scores: Tag scores and evidence
        frames: Frame data
        tag_hierarchy: Dict mapping tag_id to hierarchy info (parent_tag_ids, child_tag_ids)
    
    Returns:
        List of suggestion dictionaries with is_backup flag set for parent tags
    """
    
    if tag_hierarchy is None:
        tag_hierarchy = {}
    
    suggestions = []
    
    # Filter tags by minimum score threshold with adaptive per-tag thresholds
    # Default threshold is now 0.20 (20%) - lowered from 0.3 to allow more tags through
    # UI can filter further if needed
    
    # Tag classifications are loaded from configuration (see _get_tag_classification_for_scoring).
    # If TAG_SCORING_CLASSIFICATIONS env-var is not set, all tags default to "common".
    
    # Adaptive thresholds based on tag classification
    # Load from settings if available, otherwise use defaults
    threshold_brief = self.settings.tag_threshold_brief if hasattr(self.settings, 'tag_threshold_brief') else 0.18
    threshold_common = self.settings.tag_threshold_common if hasattr(self.settings, 'tag_threshold_common') else 0.20
    threshold_high_precision = self.settings.tag_threshold_high_precision if hasattr(self.settings, 'tag_threshold_high_precision') else 0.25
    
    THRESHOLDS = {
        "brief": threshold_brief,
        "common": threshold_common,
        "high_precision": threshold_high_precision
    }
    
    def get_tag_classification(tag_name: str) -> str:
        """Get tag classification (brief, common, or high_precision)"""
        return _get_tag_classification_for_scoring(tag_name)
    
    default_min_score = self.settings.suggestion_min_score if hasattr(self.settings, 'suggestion_min_score') else 0.20
    
    for tag_id, score_data in tag_scores.items():
        score = score_data["score"]
        tag_name = score_data["tag_name"]
        
        # Get tag classification and corresponding threshold
        tag_classification = get_tag_classification(tag_name)
        tag_threshold = THRESHOLDS[tag_classification]
        
        if score < tag_threshold:
            logger.debug(
                "Tag score below threshold",
                job_id=job_id,
                tag_name=tag_name,
                score=round(score, 3),
                threshold=tag_threshold,
                classification=tag_classification
            )
            continue
        
        # Determine confidence level
        confidence_level = _classify_confidence(self, score)
        
        # Get scene_id from first frame (all frames should have same scene_id)
        scene_id = frames[0]["scene_id"] if frames else None
        
        suggestion = {
            "id": str(uuid.uuid4()),
            "job_id": job_id,
            "scene_id": scene_id,
            "tag_id": tag_id,
            "tag_name": tag_name,
            "confidence": score,
            "confidence_level": confidence_level,
            "vision_confidence": score_data["visual_score"],
            "asr_confidence": score_data.get("asr_score", None),
            "ocr_confidence": score_data.get("ocr_score", None),
            "temporal_consistency": score_data["temporal_score"],
            "calibrated_confidence": score,  # Same as confidence for now
            "evidence_frames": score_data["evidence_frames"],
            "reasoning": f"Visual: {score_data['visual_score']:.2f}, Text: {score_data['text_score']:.2f}, Temporal: {score_data['temporal_score']:.2f}",
            "signal_details": {
                "visual_score": score_data["visual_score"],
                "text_score": score_data["text_score"],
                "temporal_score": score_data["temporal_score"],
                "match_count": score_data["match_count"],
                "total_frames": len(frames)
            },
            "is_backup": False  # Will be set by hierarchical filtering
        }
        suggestions.append(suggestion)
        
        # Log high-confidence suggestions with more detail
        if score >= 0.7:
            logger.info(
                "HIGH CONFIDENCE tag suggestion generated",
                job_id=job_id,
                tag_name=tag_name,
                confidence=round(score, 3),
                level=confidence_level,
                visual_score=round(score_data["visual_score"], 3),
                text_score=round(score_data["text_score"], 3),
                temporal_score=round(score_data["temporal_score"], 3),
                evidence_frames=len(score_data["evidence_frames"])
            )
        else:
            logger.debug(
                "Tag suggestion generated",
                job_id=job_id,
                tag_name=tag_name,
                confidence=round(score, 3),
                level=confidence_level
            )
    
    # Sort by confidence (descending)
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Apply hierarchical filtering: mark parent tags as backup if child tags exist with higher confidence
    # Build a set of all tag IDs that are in suggestions (for fast lookup)
    suggestion_tag_ids = {s["tag_id"] for s in suggestions}
    
    # For each suggestion, check if any higher-confidence tag is its descendant
    for i, suggestion in enumerate(suggestions):
        tag_id = suggestion["tag_id"]
        tag_info = tag_hierarchy.get(tag_id, {})
        child_tag_ids = tag_info.get("child_tag_ids", [])
        
        if not child_tag_ids:
            continue  # No children, cannot be a backup
        
        # Check if any higher-confidence suggestion (before this one in sorted list) is a descendant
        # A descendant is a child, grandchild, etc.
        for j in range(i):  # Only check higher-confidence tags (earlier in sorted list)
            higher_conf_tag = suggestions[j]
            higher_conf_tag_id = higher_conf_tag["tag_id"]
            
            # Check if higher_conf_tag is a descendant of tag_id
            if _is_descendant(higher_conf_tag_id, tag_id, tag_hierarchy):
                # Mark this tag as backup since a more specific (child) tag exists with higher confidence
                suggestion["is_backup"] = True
                logger.debug(
                    "Tag marked as backup (child tag has higher confidence)",
                    job_id=job_id,
                    parent_tag=suggestion["tag_name"],
                    parent_confidence=round(suggestion["confidence"], 3),
                    child_tag=higher_conf_tag["tag_name"],
                    child_confidence=round(higher_conf_tag["confidence"], 3)
                )
                break
    
    # Log backup tags
    backup_tags = [s for s in suggestions if s.get("is_backup", False)]
    if backup_tags:
        logger.info(
            "Hierarchical filtering: marked parent tags as backup",
            job_id=job_id,
            backup_count=len(backup_tags),
            backup_tags=[(s["tag_name"], round(s["confidence"], 3)) for s in backup_tags[:5]]
        )
    
    return suggestions


def _is_descendant(descendant_tag_id: str, ancestor_tag_id: str, tag_hierarchy: Dict[str, Dict[str, Any]]) -> bool:
    """
    Check if descendant_tag_id is a descendant (child, grandchild, etc.) of ancestor_tag_id
    
    Args:
        descendant_tag_id: Tag ID to check if it's a descendant
        ancestor_tag_id: Tag ID to check if it's an ancestor
        tag_hierarchy: Dict mapping tag_id to hierarchy info
    
    Returns:
        True if descendant_tag_id is a descendant of ancestor_tag_id
    """
    if descendant_tag_id == ancestor_tag_id:
        return False  # A tag is not its own descendant
    
    # Get child tags of ancestor
    ancestor_info = tag_hierarchy.get(ancestor_tag_id, {})
    child_tag_ids = ancestor_info.get("child_tag_ids", [])
    
    # Check direct children
    if descendant_tag_id in child_tag_ids:
        return True
    
    # Recursively check grandchildren, etc.
    for child_id in child_tag_ids:
        if _is_descendant(descendant_tag_id, child_id, tag_hierarchy):
            return True
    
    return False


def _classify_confidence(self, score: float) -> str:
    """
    Classify confidence score into level
    
    Args:
        score: Confidence score (0.0 to 1.0)
    
    Returns:
        Confidence level string
    """
    
    # Adjusted thresholds to better reflect actual confidence
    # High confidence: very strong match (70%+)
    if score >= 0.70:
        return "high"
    # Medium confidence: moderate match (50-70%)
    elif score >= 0.50:
        return "medium"
    # Low confidence: weak match (<50%)
    else:
        return "low"


def _count_by_confidence(self, suggestions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count suggestions by confidence level
    
    Args:
        suggestions: List of suggestions
    
    Returns:
        Dict with counts by level
    """
    
    counts = {"high": 0, "medium": 0, "low": 0}
    
    for suggestion in suggestions:
        level = suggestion.get("confidence_level", "low")
        counts[level] = counts.get(level, 0) + 1
    
    return counts


@app.task(bind=True, base=FusionTask, name='app.fusion.recompute_suggestions')
def recompute_suggestions(self, job_id: str) -> Dict[str, Any]:
    """
    Recompute suggestions for a completed job
    
    This can be used to update suggestions when tag embeddings change
    or when fusion parameters are tuned.
    
    Args:
        job_id: Job identifier
    
    Returns:
        {
            "success": bool,
            "suggestion_count": int
        }
    """
    
    logger.info("Recomputing suggestions", job_id=job_id)
    
    try:
        # Simulate embeddings and text results (already in database)
        embeddings_result = {"embedding_count": -1}  # Flag to skip check
        text_result = {"total_text_entries": -1}
        
        result = generate_suggestions(job_id, embeddings_result, text_result)
        
        logger.info(
            "Suggestions recomputed",
            job_id=job_id,
            suggestion_count=result.get("suggestion_count", 0)
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Recompute suggestions failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}
