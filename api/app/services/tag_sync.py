"""
Tag synchronization service - keeps database tags in sync with tag_prompts.json
"""

import json
import re
import structlog
from pathlib import Path
from typing import List, Dict, Any
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import async_session_maker, tags_table

logger = structlog.get_logger(__name__)


async def sync_tags_from_prompts() -> Dict[str, Any]:
    """
    Synchronize tags from tag_prompts.json into the database.
    
    This ensures all tags from tag_prompts.json are available in the database
    and keeps them in sync. Tags are marked as active if they exist in tag_prompts.json.
    
    Returns:
        Dict with sync statistics
    """
    
    # Find tag_prompts.json - try multiple locations
    import os
    custom_path = os.environ.get("TAG_PROMPTS_PATH")
    tag_prompts_paths = []
    if custom_path:
        tag_prompts_paths.append(Path(custom_path))
    tag_prompts_paths.extend([
        Path("/app/prompts/tag_prompts.json"),  # Mounted prompts directory
        Path("/app/tag_prompts.json"),  # Legacy location
        Path("/app/workers/tag_prompts.json"),  # In API container
        Path("./prompts/tag_prompts.json"),  # Relative path
    ])
    
    tag_prompts_file = None
    for path in tag_prompts_paths:
        if path.exists():
            tag_prompts_file = path
            break
    
    if not tag_prompts_file:
        logger.warning("tag_prompts.json not found, skipping tag sync")
        return {
            "success": False,
            "error": "tag_prompts.json not found",
            "tags_created": 0,
            "tags_updated": 0
        }
    
    try:
        with open(tag_prompts_file, 'r', encoding='utf-8') as f:
            tag_prompts = json.load(f)
    except Exception as e:
        logger.error("Failed to load tag_prompts.json", error=str(e))
        return {
            "success": False,
            "error": f"Failed to load tag_prompts.json: {str(e)}",
            "tags_created": 0,
            "tags_updated": 0
        }
    
    logger.info("Syncing tags from tag_prompts.json", tag_count=len(tag_prompts))
    
    tags_created = 0
    tags_updated = 0
    tags_skipped = 0
    
    async with async_session_maker() as session:
        async with session.begin():
            for tag_name, prompts in tag_prompts.items():
                if not prompts or not isinstance(prompts, list) or len(prompts) == 0:
                    tags_skipped += 1
                    continue
                
                # Generate tag_id from name (sanitize)
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
                        # Fallback: access by column name
                        existing_dict = {
                            'tag_id': existing.tag_id,
                            'name': existing.name
                        }
                    
                    # Update existing tag to ensure it's active and sync prompts
                    update_query = sa.update(tags_table).where(
                        tags_table.c.tag_id == existing_dict['tag_id']
                    ).values(
                        name=tag_name,
                        is_active=True,
                        prompts=prompts  # Store prompts in database
                    )
                    await session.execute(update_query)
                    tags_updated += 1
                else:
                    # Create new tag with prompts
                    insert_query = sa.insert(tags_table).values(
                        tag_id=tag_id,
                        name=tag_name,
                        is_active=True,
                        prompts=prompts  # Store prompts in database
                    )
                    await session.execute(insert_query)
                    tags_created += 1
            
            # Mark tags not in tag_prompts.json as inactive
            # Get all tag names from tag_prompts.json
            tag_names_in_prompts = set(tag_prompts.keys())
            
            # Get all active tags from database
            all_tags_query = sa.select(tags_table.c.tag_id, tags_table.c.name).where(
                tags_table.c.is_active == True
            )
            result = await session.execute(all_tags_query)
            all_tags = result.fetchall()
            
            tags_deactivated = 0
            for tag_row in all_tags:
                # Convert SQLAlchemy Row to dict
                if hasattr(tag_row, '_mapping'):
                    tag_dict = dict(tag_row._mapping)
                elif hasattr(tag_row, '_asdict'):
                    tag_dict = tag_row._asdict()
                else:
                    # Fallback: access by column name
                    tag_dict = {
                        'tag_id': tag_row.tag_id,
                        'name': tag_row.name
                    }
                
                if tag_dict['name'] not in tag_names_in_prompts:
                    # Tag exists in DB but not in tag_prompts.json - deactivate it
                    deactivate_query = sa.update(tags_table).where(
                        tags_table.c.tag_id == tag_dict['tag_id']
                    ).values(is_active=False)
                    await session.execute(deactivate_query)
                    tags_deactivated += 1
    
    logger.info(
        "Tag sync completed",
        tags_created=tags_created,
        tags_updated=tags_updated,
        tags_skipped=tags_skipped,
        tags_deactivated=tags_deactivated,
        total_in_prompts=len(tag_prompts)
    )
    
    return {
        "success": True,
        "tags_created": tags_created,
        "tags_updated": tags_updated,
        "tags_skipped": tags_skipped,
        "tags_deactivated": tags_deactivated,
        "total_in_prompts": len(tag_prompts)
    }


async def export_tags_to_prompts() -> Dict[str, Any]:
    """
    Export tags with prompts from database to tag_prompts.json format.
    
    Returns:
        Dict with tag prompts in the format expected by tag_prompts.json
    """
    try:
        async with async_session_maker() as session:
            query = sa.select(tags_table).where(tags_table.c.is_active == True)
            result = await session.execute(query)
            tags = result.fetchall()
            
            tag_prompts = {}
            for tag in tags:
                if hasattr(tag, '_mapping'):
                    tag_dict = dict(tag._mapping)
                elif hasattr(tag, '_asdict'):
                    tag_dict = tag._asdict()
                else:
                    tag_dict = {
                        'name': tag.name,
                        'prompts': tag.prompts if hasattr(tag, 'prompts') else []
                    }
                
                tag_name = tag_dict.get('name')
                prompts = tag_dict.get('prompts', [])
                
                if tag_name and prompts:
                    tag_prompts[tag_name] = prompts
            
            return {
                "success": True,
                "tag_prompts": tag_prompts,
                "count": len(tag_prompts)
            }
            
    except Exception as e:
        logger.error("Failed to export tags to prompts", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "tag_prompts": {},
            "count": 0
        }

