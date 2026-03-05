"""
Embeddings Worker - Generates visual embeddings using CLIP ViT-L/14

This worker processes extracted frames and generates dense embeddings 
for visual content understanding using OpenAI's CLIP model.
"""

import uuid
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from celery import Task
import structlog
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPModel
try:
    from transformers import SiglipProcessor, SiglipModel
    SIGLIP_AVAILABLE = True
except ImportError:
    SIGLIP_AVAILABLE = False
    SiglipProcessor = None
    SiglipModel = None

from .celery_app import app
from .database import get_database_connection, store_embeddings, get_job_frames
from .config import get_worker_settings

logger = structlog.get_logger(__name__)


class CLIPEmbedder:
    """CLIP ViT-L/14 visual-text embedder for content tagging"""
    
    def __init__(self, model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading vision model {model_name} on {self.device}")
        
        # Detect model type and load appropriate processor/model
        is_siglip = "siglip" in model_name.lower()
        
        if is_siglip:
            # SigLIP models use SiglipProcessor and SiglipModel
            if not SIGLIP_AVAILABLE:
                raise ImportError(
                    f"SigLIP model requested ({model_name}) but SiglipProcessor/SiglipModel not available. "
                    f"Please ensure transformers library is up to date: pip install --upgrade transformers"
                )
            logger.info(f"Detected SigLIP model, using SiglipProcessor and SiglipModel")
            self.processor = SiglipProcessor.from_pretrained(
                model_name,
                cache_dir="/tmp/cache/transformers",
                trust_remote_code=True
            )
            self.model = SiglipModel.from_pretrained(
                model_name,
                cache_dir="/tmp/cache/transformers",
                trust_remote_code=True,
                use_safetensors=True
            ).to(self.device)
        else:
            # CLIP models use AutoProcessor and CLIPModel
            logger.info(f"Detected CLIP model, using AutoProcessor and CLIPModel")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir="/tmp/cache/transformers",
                trust_remote_code=True
            )
            self.model = CLIPModel.from_pretrained(
                model_name,
                cache_dir="/tmp/cache/transformers",
                trust_remote_code=True,
                use_safetensors=True
            ).to(self.device)
        
        self.model.eval()
        self.is_siglip = is_siglip
        
        # Get target image size from processor config
        self.image_size = getattr(self.processor.image_processor, 'size', {}).get('shortest_edge', 224)
        if isinstance(self.image_size, dict):
            self.image_size = self.image_size.get('shortest_edge', 224)
        
        logger.info(f"CLIP model configured with image size: {self.image_size}")
        
        # Dynamically detect embedding dimension from model
        # Test with a dummy input to get actual output dimension
        try:
            with torch.no_grad():
                # Create a dummy image using PIL and process it through the processor
                from PIL import Image
                dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
                dummy_inputs = self.processor(images=[dummy_image], return_tensors="pt")
                dummy_pixel_values = dummy_inputs['pixel_values'].to(self.device)
                dummy_features = self.model.get_image_features(pixel_values=dummy_pixel_values)
                self.embedding_dim = int(dummy_features.shape[-1])
                logger.info(f"Detected embedding dimension: {self.embedding_dim} for model {model_name}")
        except Exception as e:
            # Fallback: try to get from model config or use defaults based on model name
            logger.warning(f"Could not detect embedding dimension, using model-specific defaults: {e}")
            if "ViT-H" in model_name or "vit-h" in model_name.lower():
                self.embedding_dim = 1024  # ViT-H models typically use 1024
            elif "siglip" in model_name.lower() or "so400m" in model_name.lower():
                self.embedding_dim = 1152  # SigLIP models typically use 1152
            elif "ViT-L" in model_name or "vit-l" in model_name.lower():
                self.embedding_dim = 768  # ViT-L models use 768
            elif "ViT-B" in model_name or "vit-b" in model_name.lower():
                self.embedding_dim = 512  # ViT-B models use 512
            else:
                self.embedding_dim = 768  # Default fallback
            logger.info(f"Using default embedding dimension: {self.embedding_dim} for model {model_name}")
        
        # Load tag prompts for embeddings
        self.tag_prompts = self._load_tag_prompts()
        
        model_type = "SigLIP" if self.is_siglip else "CLIP"
        logger.info(f"{model_type} embedder ready with {len(self.tag_prompts)} tag prompts, embedding_dim={self.embedding_dim}")
    
    def _load_tag_prompts(self):
        """Load tag visual descriptions"""
        prompts_file = Path(os.environ.get("TAG_PROMPTS_PATH", "/app/prompts/tag_prompts.json"))
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                    logger.info(f"Loaded {len(prompts)} tag prompts from {prompts_file}")
                    return prompts
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse tag_prompts.json: {e}. "
                    f"File may be corrupted. Using empty prompts.",
                    exc_info=True
                )
                return {}
            except Exception as e:
                logger.error(
                    f"Error loading tag_prompts.json: {e}. Using empty prompts.",
                    exc_info=True
                )
                return {}
        else:
            logger.warning("tag_prompts.json not found, using empty prompts")
            return {}
    
    def reload_tag_prompts(self):
        """Reload tag prompts from file (useful for hot-reloading during development)"""
        self.tag_prompts = self._load_tag_prompts()
        logger.info(f"Reloaded {len(self.tag_prompts)} tag prompts")
    
    @torch.no_grad()
    def encode_images_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Encode batch of images using CLIP or SigLIP model
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of normalized image embedding vectors (dimension varies by model)
        """
        # Load images
        images = []
        valid_indices = []
        
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
        
        if not images:
            return []
        
        # Try batch processing first (much more efficient)
        try:
            # Process all images in one batch
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            )
            
            # Move to device
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # Get image embeddings in batch
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            
            # Normalize embeddings
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and move to CPU immediately
            embeddings_np = image_embeddings.cpu().numpy()
            
            # Clear GPU memory
            del pixel_values
            del image_features
            del image_embeddings
            del inputs
            
            # Clear image data from memory
            for img in images:
                img.close()
            del images
            
            # Force garbage collection if using CUDA
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            
            logger.debug(f"Generated {len(embeddings_np)} CLIP embeddings in batch from {len(image_paths)} images")
            return [emb for emb in embeddings_np]
            
        except Exception as e:
            # Fallback to one-at-a-time if batch fails
            logger.warning(f"Batch processing failed: {e}, falling back to individual processing")
            all_embeddings = []
            
            for i, img in enumerate(images):
                try:
                    inputs = self.processor(
                        images=[img],
                        return_tensors="pt"
                    )
                    
                    pixel_values = inputs['pixel_values'].to(self.device)
                    image_features = self.model.get_image_features(pixel_values=pixel_values)
                    image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                    embedding_np = image_embedding.cpu().numpy()[0]
                    all_embeddings.append(embedding_np)
                    
                    # Clear GPU memory after each image in fallback mode
                    del pixel_values
                    del image_features
                    del image_embedding
                    del inputs
                    img.close()
                    
                    if self.device == "cuda":
                        import torch
                        torch.cuda.empty_cache()
                    
                except Exception as img_error:
                    logger.error(f"Failed to process image {i}: {img_error}")
                    all_embeddings.append(None)  # Keep alignment with input
                    if img:
                        img.close()
                    continue
            
            # Clear remaining images
            for img in images:
                if img:
                    img.close()
            del images
            
            logger.debug(f"Generated {len([e for e in all_embeddings if e is not None])} CLIP embeddings (fallback mode)")
            return all_embeddings
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text string using CLIP or SigLIP"""
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move inputs to device - handle both CLIP (has attention_mask) and SigLIP (only input_ids)
        inputs_to_device = {}
        for key, value in inputs.items():
            inputs_to_device[key] = value.to(self.device)
        
        # Get text features - use **inputs to handle both CLIP and SigLIP automatically
        text_features = self.model.get_text_features(**inputs_to_device)
        
        embedding = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0]
    
    def compute_image_text_similarity(self, image_path: str, text: str) -> float:
        """Compute similarity between image and text using CLIP or SigLIP"""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Process both image and text with CLIP or SigLIP
            inputs = self.processor(
                images=[image],
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move inputs to device - handle both CLIP (has attention_mask) and SigLIP (only input_ids)
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # For text, move all text-related inputs to device
            text_inputs = {}
            for key in ['input_ids', 'attention_mask']:
                if key in inputs:
                    text_inputs[key] = inputs[key].to(self.device)
            
            # Get embeddings - use **inputs to handle both CLIP and SigLIP automatically
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (cosine similarity via dot product)
            similarity = torch.mm(image_features, text_features.t()).item()
            
            # CLIP outputs are already in [-1, 1] range, convert to [0, 1]
            return (similarity + 1.0) / 2.0
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0


class EmbeddingsTask(Task):
    """Base task with shared CLIP model instance"""
    
    _embedder = None
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
    
    def _get_clip_model_name_from_db(self) -> str:
        """Get clip_model_name from database settings, fallback to config/env"""
        default_model = self.settings.clip_model_name if hasattr(self.settings, 'clip_model_name') else "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        
        try:
            # Try to get from database settings table
            import asyncio
            import asyncpg
            from .database import run_async
            
            async def get_db_setting():
                db_url = self.settings.database_url
                if db_url.startswith("postgresql+asyncpg://"):
                    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
                
                conn = await asyncpg.connect(db_url)
                try:
                    # Query settings table for clip_model_name
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
            
            db_value = run_async(get_db_setting())
            if db_value:
                logger.debug(f"Found clip_model_name in database: {db_value}")
                return db_value
        except Exception as e:
            logger.debug(f"Could not read clip_model_name from database, using default: {e}")
        
        return default_model
    
    @property
    def embedder(self):
        """Get CLIP embedder (cached and shared across tasks)"""
        # Get model name from database settings (with fallback to config/env)
        model_name = self._get_clip_model_name_from_db()
        
        # Check if embedder needs to be reloaded (model changed or not initialized)
        if self._embedder is None or (hasattr(self._embedder, 'model_name') and self._embedder.model_name != model_name):
            if self._embedder is not None:
                logger.info(f"Model changed from {self._embedder.model_name} to {model_name}, reloading CLIP embedder")
            else:
                logger.info(f"Loading CLIP embedder with model {model_name}")
            self._embedder = CLIPEmbedder(model_name=model_name)
            logger.info("CLIP embedder loaded successfully")
        return self._embedder
    
    def _process_frame_batch_sync(self, job_id: str, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of frames to generate embeddings using CLIP (sync version for executor)
        
        Args:
            job_id: Job identifier
            frames: List of frame metadata with file_path
        
        Returns:
            List of embedding data dictionaries
        """
        
        embeddings_data = []
        
        # Validate frame files exist
        valid_frames = []
        for frame in frames:
            frame_path = frame.get("file_path")
            if not frame_path:
                logger.warning(
                    "Frame missing file_path",
                    job_id=job_id,
                    frame_id=frame.get("id")
                )
                continue
            
            # Try multiple possible paths for frame file
            # The sampler stores frames at /app/.cache/frames/{job_id}/frame_{number}.jpg
            # The embeddings worker can access them at /app/.cache/frames/{job_id}/frame_{number}.jpg
            actual_path = None
            if Path(frame_path).exists():
                actual_path = frame_path
            else:
                # Try alternative paths based on job_id and frame_number
                frame_job_id = frame.get("job_id") or job_id
                frame_number = frame.get("frame_number", 0)
                if frame_job_id:
                    # Try shared volume paths
                    alt_paths = [
                        f"/app/.cache/frames/{frame_job_id}/frame_{frame_number:06d}.jpg",
                        f"/app/frames/{frame_job_id}/frame_{frame_number:06d}.jpg",
                    ]
                    for alt_path in alt_paths:
                        if Path(alt_path).exists():
                            actual_path = alt_path
                            break
            
            if not actual_path:
                logger.warning(
                    "Frame file not found",
                    job_id=job_id,
                    frame_id=frame.get("id"),
                    original_path=frame_path,
                    frame_number=frame.get("frame_number")
                )
                continue
            
            # Update frame dict with actual path
            frame["file_path"] = actual_path
            valid_frames.append(frame)
        
        if not valid_frames:
            return []
        
        try:
            # Extract image paths
            image_paths = [frame["file_path"] for frame in valid_frames]
            
            # Generate embeddings in batch using CLIP
            embeddings = self.embedder.encode_images_batch(image_paths)
            
            # Prepare data for storage - handle None embeddings from failed processing
            for frame, embedding in zip(valid_frames, embeddings):
                if embedding is None:
                    logger.warning(f"Skipping frame {frame.get('id')} - embedding generation failed")
                    continue
                    
                embeddings_data.append({
                    "id": str(uuid.uuid4()),
                    "frame_id": frame["id"],
                    "model_name": self.embedder.model_name,
                    "embedding": embedding if isinstance(embedding, np.ndarray) else np.array(embedding),
                    "metadata": {
                        "frame_number": frame.get("frame_number"),
                        "timestamp": frame.get("timestamp_seconds"),
                        "width": frame.get("width"),
                        "height": frame.get("height"),
                        "embedding_dim": self.embedder.embedding_dim
                    }
                })
            
            # Clear embeddings from memory after copying to storage format
            del embeddings
            del image_paths
            
            logger.debug(
                "Batch CLIP embeddings generated",
                job_id=job_id,
                batch_size=len(embeddings_data)
            )
            
        except Exception as e:
            logger.error(
                "Batch CLIP embedding generation failed",
                job_id=job_id,
                batch_size=len(valid_frames),
                error=str(e),
                exc_info=True
            )
            # Continue processing - partial results are acceptable
        
        return embeddings_data



@app.task(bind=True, base=EmbeddingsTask, name='app.embeddings.generate_embeddings')
def generate_embeddings(self, job_id: str, frame_data) -> Dict[str, Any]:
    """
    Generate CLIP embeddings for extracted frames
    
    Args:
        job_id: Unique job identifier
        frame_data: Result from sampler (dict) or frame_ids (list) - for compatibility
    
    Returns:
        {
            "success": bool,
            "embedding_count": int,
            "model_name": str,
            "embedding_dim": int
        }
    """
    
    # Handle both dict and list inputs for backward compatibility
    if isinstance(frame_data, dict):
        frame_count = frame_data.get("frame_count", 0)
    else:
        frame_count = len(frame_data) if isinstance(frame_data, list) else 0
    
    logger.info(
        "Starting CLIP embeddings generation",
        job_id=job_id,
        frame_count=frame_count,
        input_type=type(frame_data).__name__
    )
    
    try:
        # Use asyncio.run to handle async database calls
        import asyncio
        import asyncpg
        import json
        from .config import get_worker_settings
        
        async def process_embeddings():
            # Create direct database connection to avoid pool conflicts
            settings = get_worker_settings()
            db_url = settings.database_url
            
            # Convert SQLAlchemy URL to asyncpg format if needed
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            
            conn = await asyncpg.connect(db_url)
            
            try:
                # Get frame information from database
                frames_query = """
                    SELECT id, job_id, scene_id, frame_number, timestamp_seconds, file_path, width, height
                    FROM frame_samples
                    WHERE job_id = $1
                    ORDER BY frame_number
                """
                
                frames_records = await conn.fetch(frames_query, job_id)
                frames = [dict(record) for record in frames_records]
                
                if not frames:
                    logger.warning("No frames found for job", job_id=job_id)
                    # Get default model name and dimension from settings
                    default_model = self.settings.clip_model_name if hasattr(self.settings, 'clip_model_name') else "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
                    return {
                        "success": True,
                        "embedding_count": 0,
                        "model_name": default_model,
                        "embedding_dim": 768  # Default fallback
                    }
                
                logger.info(
                    "Retrieved frames from database",
                    job_id=job_id,
                    frame_count=len(frames)
                )
                
                # Process frames in batches for efficiency
                # Use smaller batch size to prevent OOM - CLIP ViT-L/14 is memory intensive
                batch_size = getattr(self.settings, 'embedding_batch_size', None) or getattr(self.settings, 'batch_size', 4)
                if batch_size > 8:
                    batch_size = 4  # Cap at 4 for safety
                    logger.warning("Capping embedding batch size at 4 to prevent OOM", job_id=job_id)
                
                embeddings_data = []
                total_batches = (len(frames) + batch_size - 1) // batch_size
                
                logger.info(
                    "Processing frames in batches",
                    job_id=job_id,
                    total_frames=len(frames),
                    batch_size=batch_size,
                    total_batches=total_batches
                )
                
                for i in range(0, len(frames), batch_size):
                    batch_frames = frames[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    logger.debug(
                        "Processing embeddings batch",
                        job_id=job_id,
                        batch_num=batch_num,
                        total_batches=total_batches,
                        batch_size=len(batch_frames),
                        frame_numbers=[f.get("frame_number") for f in batch_frames]
                    )
                    
                    # Run CPU-intensive embedding generation in thread pool
                    import concurrent.futures
                    import functools
                    
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        batch_embeddings = await loop.run_in_executor(
                            executor, 
                            functools.partial(self._process_frame_batch_sync, job_id, batch_frames)
                        )
                    
                    embeddings_data.extend(batch_embeddings)
                    
                    # Force garbage collection after each batch to free memory
                    import gc
                    gc.collect()
                    
                    logger.info(
                        "Processed embeddings batch",
                        job_id=job_id,
                        batch_num=batch_num,
                        total_batches=total_batches,
                        batch_size=len(batch_frames),
                        embeddings_generated=len(batch_embeddings),
                        total_embeddings_so_far=len(embeddings_data)
                    )
                
                # Store embeddings in database using batch insert for efficiency
                if embeddings_data:
                    # Use transaction with batch insert for efficiency
                    async with conn.transaction():
                        store_query = """
                            INSERT INTO embeddings (id, frame_id, model_name, embedding, metadata, created_at)
                            VALUES ($1, $2, $3, $4::vector, $5, NOW())
                        """
                        
                        # Prepare and execute batch inserts
                        for embedding_data in embeddings_data:
                            # Convert embedding to pgvector string format
                            embedding_list = embedding_data["embedding"]
                            if isinstance(embedding_list, np.ndarray):
                                embedding_list = embedding_list.tolist()
                            
                            # Convert list to pgvector string format: '[1.0,2.0,3.0]'
                            embedding_str = '[' + ','.join(str(float(x)) for x in embedding_list) + ']'
                            
                            await conn.execute(
                                store_query,
                                embedding_data["id"],
                                embedding_data["frame_id"],
                                embedding_data["model_name"],
                                embedding_str,
                                json.dumps(embedding_data["metadata"])
                            )
                
                logger.info(
                    "CLIP embeddings generation completed",
                    job_id=job_id,
                    embedding_count=len(embeddings_data),
                    model_name=self.embedder.model_name,
                    embedding_dim=self.embedder.embedding_dim
                )
                
                return {
                    "success": True,
                    "embedding_count": len(embeddings_data),
                    "model_name": self.embedder.model_name,
                    "embedding_dim": self.embedder.embedding_dim  # Dynamically detected dimension
                }
                
            finally:
                await conn.close()
        
        # Run the async processing
        from .database import run_async
        return run_async(process_embeddings())
        
    except Exception as e:
        logger.error(
            "CLIP embeddings generation failed",
            job_id=job_id,
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}


@app.task(bind=True, base=EmbeddingsTask, name='app.embeddings.generate_tag_embeddings')
def generate_tag_embeddings(self, tag_names: List[str]) -> Dict[str, Any]:
    """
    Generate text embeddings for tags using CLIP
    
    Args:
        tag_names: List of tag names to generate embeddings for
    
    Returns:
        {
            "success": bool,
            "embeddings": Dict[str, List[float]]  # tag_name -> embedding
        }
    """
    
    logger.info(
        "Generating CLIP tag embeddings",
        tag_count=len(tag_names)
    )
    
    try:
        embeddings = {}
        
        for tag_name in tag_names:
            # Use first prompt for the tag if available (case-insensitive match)
            tag_prompt_key = None
            for key in self.embedder.tag_prompts.keys():
                if key.lower() == tag_name.lower():
                    tag_prompt_key = key
                    break
            
            if tag_prompt_key and self.embedder.tag_prompts[tag_prompt_key]:
                text = self.embedder.tag_prompts[tag_prompt_key][0]
                logger.debug(
                    "Using tag prompt for embedding",
                    tag_name=tag_name,
                    prompt_key=tag_prompt_key,
                    prompt_text=text[:50] + "..." if len(text) > 50 else text
                )
            else:
                text = tag_name  # Fallback to tag name
                logger.debug(
                    "No tag prompt found, using tag name",
                    tag_name=tag_name
                )
                
            # Generate text embedding
            embedding = self.embedder.encode_text(text)
            embeddings[tag_name] = embedding.tolist()
            
            logger.debug(
                "Tag embedding generated",
                tag_name=tag_name,
                prompt_used=text,
                embedding_dim=len(embedding)
            )
        
        logger.info(
            "CLIP tag embeddings generation completed",
            tag_count=len(embeddings)
        )
        
        return {
            "success": True,
            "embeddings": embeddings,
            "embedding_dim": self.embedder.embedding_dim
        }
        
    except Exception as e:
        logger.error(
            "CLIP tag embeddings generation failed",
            error=str(e),
            exc_info=True
        )
        return {"success": False, "error": str(e)}


@app.task(name='app.embeddings.compute_similarity')
def compute_similarity(
    image_embedding: List[float],
    text_embedding: List[float]
) -> float:
    """
    Compute cosine similarity between image and text embeddings
    
    Args:
        image_embedding: Image embedding vector
        text_embedding: Text embedding vector
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    
    import numpy as np
    
    # Convert to numpy arrays
    img_emb = np.array(image_embedding)
    txt_emb = np.array(text_embedding)
    
    # Compute cosine similarity
    # Normalize embeddings
    img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)
    txt_emb = txt_emb / (np.linalg.norm(txt_emb) + 1e-8)
    
    # Dot product gives cosine similarity
    similarity = float(np.dot(img_emb, txt_emb))
    
    # Ensure in valid range [0, 1]
    similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    return similarity


@app.task(bind=True, base=EmbeddingsTask, name='app.embeddings.analyze_frame_for_tags') 
def analyze_frame_for_tags(self, frame_path: str, tag_names: List[str]) -> Dict[str, float]:
    """
    Analyze a single frame against specific tags using CLIP
    
    Args:
        frame_path: Path to image frame
        tag_names: List of tag names to analyze against
        
    Returns:
        Dict mapping tag names to similarity scores
    """
    
    logger.info(
        "Analyzing frame for tags",
        frame_path=frame_path,
        tag_count=len(tag_names)
    )
    
    try:
        results = {}
        
        for tag_name in tag_names:
            # Get prompts for this tag
            prompts = self.embedder.tag_prompts.get(tag_name, [tag_name])
            
            max_similarity = 0.0
            for prompt in prompts:
                similarity = self.embedder.compute_image_text_similarity(frame_path, prompt)
                max_similarity = max(max_similarity, similarity)
            
            results[tag_name] = max_similarity
            
            logger.debug(
                "Tag analysis complete",
                tag_name=tag_name,
                similarity=max_similarity,
                frame_path=frame_path
            )
        
        return results
        
    except Exception as e:
        logger.error(
            "Frame tag analysis failed",
            frame_path=frame_path,
            error=str(e),
            exc_info=True
        )
        return {tag: 0.0 for tag in tag_names}
