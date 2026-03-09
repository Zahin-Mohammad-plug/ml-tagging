"""Service for generating tag suggestions based on video descriptions and titles using CLIP text embeddings"""

from typing import Optional, List, Dict, Any
import structlog
import sqlalchemy as sa
import numpy as np
import json
import sys
from pathlib import Path

# Add workers directory to path to import CLIPEmbedder
workers_path = Path(__file__).parent.parent.parent.parent / "workers"
if str(workers_path) not in sys.path:
    sys.path.insert(0, str(workers_path))

try:
    from app.embeddings import CLIPEmbedder
except ImportError:
    # Fallback: try direct import if workers is in path
    try:
        from workers.app.embeddings import CLIPEmbedder
    except ImportError:
        CLIPEmbedder = None

from ..database import async_session_maker, tags_table, text_analysis_table, settings_table
from ..config import get_settings
from ..models import VideoScene

logger = structlog.get_logger()
settings = get_settings()


class TextTagSuggester:
    """Service for generating tag suggestions from text descriptions and titles"""
    
    def __init__(self):
        self._embedder = None
        self._model_name = None
    
    async def _get_embedder(self):
        """Get or create CLIP embedder instance (lazy loading)"""
        if self._embedder is None:
            if CLIPEmbedder is None:
                raise RuntimeError(
                    "CLIPEmbedder not available. Ensure workers/app/embeddings.py is accessible."
                )
            
            # Get model name from settings
            try:
                # Try to get from database settings first
                result = await self._execute_query(
                    sa.select(settings_table.c.value).where(
                        settings_table.c.key == "clip_model_name",
                        settings_table.c.user_id.is_(None)
                    ),
                    fetch="one"
                )
                if result and result.get("value"):
                    value = result["value"]
                    # Handle JSONB - it might be a dict or string
                    if isinstance(value, dict):
                        model_name = value.get("value") or str(value)
                    elif isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            model_name = parsed.get("value") if isinstance(parsed, dict) else parsed
                        except:
                            model_name = value
                    else:
                        model_name = str(value)
                else:
                    model_name = getattr(settings, 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            except Exception as e:
                logger.warning("Could not get model name from database, using default", error=str(e))
                model_name = getattr(settings, 'clip_model_name', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            
            self._model_name = model_name
            
            # Get device setting
            device = getattr(settings, 'vision_device', 'auto')
            if device == 'auto':
                device = None  # Let CLIPEmbedder auto-detect
            
            try:
                self._embedder = CLIPEmbedder(model_name=model_name, device=device)
                logger.info("Loaded CLIP embedder for text-based suggestions", model_name=model_name)
            except Exception as e:
                logger.error("Failed to load CLIP embedder", error=str(e))
                raise
        
        return self._embedder
    
    async def _execute_query(self, query, *, fetch: Optional[str] = None, commit: bool = False):
        """Execute a query using a dedicated async session."""
        async with async_session_maker() as session:
            try:
                result = await session.execute(query)
                
                if fetch == "one":
                    row = result.mappings().first()
                    if row:
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
            except Exception as e:
                await session.rollback()
                logger.error("Query execution failed", error=str(e), exc_info=True)
                raise
    
    async def _get_ocr_text(self, scene_id: str) -> Optional[str]:
        """Get OCR text from text_analysis table for a scene"""
        try:
            # Find job_id for this scene
            from ..database import jobs_table, frame_samples_table
            query = (
                sa.select(text_analysis_table.c.text_content)
                .select_from(
                    text_analysis_table.join(
                        frame_samples_table,
                        text_analysis_table.c.frame_id == frame_samples_table.c.id
                    )
                )
                .where(
                    frame_samples_table.c.scene_id == scene_id,
                    text_analysis_table.c.analysis_type == "ocr"
                )
                .order_by(text_analysis_table.c.created_at.desc())
                .limit(50)  # Get up to 50 OCR text entries
            )
            
            rows = await self._execute_query(query, fetch="all")
            if rows:
                # Combine all OCR text
                ocr_texts = [row["text_content"] for row in rows if row.get("text_content")]
                return " ".join(ocr_texts) if ocr_texts else None
            return None
        except Exception as e:
            logger.warning("Failed to get OCR text", scene_id=scene_id, error=str(e))
            return None
    
    async def suggest_tags(
        self,
        scene: VideoScene,
        existing_tag_ids: Optional[List[str]] = None,
        use_description: bool = True,
        use_title: bool = True,
        use_ocr: bool = False,
        min_confidence: float = 0.3,
        max_suggestions: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Generate tag suggestions based on video description, title, and optionally OCR text.
        
        Args:
            scene: VideoScene object with description and title
            existing_tag_ids: List of tag IDs already on the video (to exclude)
            use_description: Whether to use video description
            use_title: Whether to use video title
            use_ocr: Whether to include OCR text if available
            min_confidence: Minimum confidence threshold for suggestions
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of suggestion dicts with keys: tag_id, tag_name, confidence, source
        """
        try:
            # Build text to encode
            text_parts = []
            if use_title and scene.title:
                text_parts.append(scene.title)
            if use_description and scene.description:
                text_parts.append(scene.description)
            
            # Optionally add OCR text
            if use_ocr:
                ocr_text = await self._get_ocr_text(scene.id)
                if ocr_text:
                    text_parts.append(ocr_text)
            
            if not text_parts:
                logger.warning("No text available for encoding", scene_id=scene.id)
                return []
            
            # Combine text parts
            combined_text = ". ".join(text_parts)
            
            # Get CLIP embedder
            embedder = await self._get_embedder()
            
            # Encode text
            logger.info("Encoding text for tag suggestions", scene_id=scene.id, text_length=len(combined_text))
            text_embedding = embedder.encode_text(combined_text)
            
            # Get all active tags with embeddings
            # Cast embedding to text for easier parsing (similar to fusion.py)
            query = (
                sa.select(
                    tags_table.c.tag_id,
                    tags_table.c.name,
                    sa.cast(tags_table.c.embedding, sa.Text).label("embedding"),
                    tags_table.c.description
                )
                .where(tags_table.c.is_active == True)
                .where(tags_table.c.embedding.is_not(None))
            )
            
            tag_records = await self._execute_query(query, fetch="all")
            
            if not tag_records:
                logger.warning("No tags with embeddings found")
                return []
            
            # Get existing tag IDs if not provided
            if existing_tag_ids is None:
                existing_tag_ids = []
                # Try to get from scene tags if available
                if scene.tags:
                    # Query tag IDs by name from our database
                    tag_names_lower = [t.lower() for t in scene.tags]
                    tag_id_query = sa.select(tags_table.c.tag_id, tags_table.c.name).where(
                        sa.func.lower(tags_table.c.name).in_(tag_names_lower)
                    )
                    tag_id_rows = await self._execute_query(tag_id_query, fetch="all")
                    existing_tag_ids = [row["tag_id"] for row in (tag_id_rows or [])]
            
            # Convert existing_tag_ids to set for fast lookup
            existing_tag_set = set(existing_tag_ids) if existing_tag_ids else set()
            
            # Compute similarities
            suggestions = []
            for tag_record in tag_records:
                tag_id = tag_record["tag_id"]
                
                # Skip existing tags
                if tag_id in existing_tag_set:
                    continue
                
                # Parse tag embedding
                tag_embedding = tag_record.get("embedding")
                if tag_embedding is None:
                    continue
                
                # Handle different embedding formats
                try:
                    if isinstance(tag_embedding, str):
                        # Try to parse as JSON array
                        try:
                            tag_emb = json.loads(tag_embedding)
                        except json.JSONDecodeError:
                            # Try to parse as pgvector string format
                            tag_emb = [float(x.strip()) for x in tag_embedding.strip('[]').split(',') if x.strip()]
                    elif hasattr(tag_embedding, 'tolist'):
                        tag_emb = tag_embedding.tolist()
                    elif isinstance(tag_embedding, (list, tuple)):
                        tag_emb = list(tag_embedding)
                    else:
                        logger.warning("Unknown embedding format", tag_id=tag_id, embedding_type=type(tag_embedding))
                        continue
                    
                    # Convert to numpy array
                    tag_emb_np = np.array(tag_emb, dtype=np.float32)
                    
                    # Compute cosine similarity
                    similarity = float(np.dot(text_embedding, tag_emb_np))
                    
                    # Filter by min_confidence
                    if similarity >= min_confidence:
                        suggestions.append({
                            "tag_id": tag_id,
                            "tag_name": tag_record["name"],
                            "confidence": similarity,
                            "source": "text",
                            "text_type": "description" if use_description else "title",
                        })
                except Exception as e:
                    logger.warning("Failed to process tag embedding", tag_id=tag_id, error=str(e))
                    continue
            
            # Sort by confidence (descending) and limit
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            suggestions = suggestions[:max_suggestions]
            
            logger.info(
                "Generated text-based tag suggestions",
                scene_id=scene.id,
                suggestion_count=len(suggestions),
                min_confidence=min_confidence
            )
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to generate text-based suggestions", scene_id=scene.id, error=str(e), exc_info=True)
            return []

