"""Service for managing tags, prompts, and blacklist"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
import structlog
import sqlalchemy as sa
from sqlalchemy import or_, func
from ..database import async_session_maker, tags_table, blacklisted_tags_table

logger = structlog.get_logger()


class TagsService:
    """Service for managing tags, prompts, and blacklist"""

    async def _execute_query(self, query, *, fetch: Optional[str] = None, commit: bool = False):
        """Execute a query using a dedicated async session."""
        async with async_session_maker() as session:
            try:
                result = await session.execute(query)
                if commit:
                    await session.commit()
                if fetch == "one":
                    return result.first()
                elif fetch == "all":
                    return result.fetchall()
                elif fetch == "scalar":
                    return result.scalar()
                return result
            except Exception:
                await session.rollback()
                raise

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert SQLAlchemy Row to dictionary with proper serialization"""
        import json
        
        # Build dictionary by accessing each column individually
        result = {}
        
        # Try to convert row using multiple methods
        row_dict = {}
        
        # Method 1: Try _mapping (SQLAlchemy 2.0+)
        if hasattr(row, '_mapping'):
            try:
                # _mapping is a MappingView, convert it properly
                mapping = row._mapping
                if hasattr(mapping, 'items'):
                    row_dict = dict(mapping.items())
                elif hasattr(mapping, 'keys'):
                    row_dict = {key: mapping[key] for key in mapping.keys()}
                else:
                    row_dict = dict(mapping)
            except Exception as e:
                logger.debug(f"Failed to convert _mapping", error=str(e))
        
        # Method 2: Try _asdict (SQLAlchemy 1.4+)
        if not row_dict and hasattr(row, '_asdict'):
            try:
                row_dict = row._asdict()
            except Exception as e:
                logger.debug(f"Failed to convert _asdict", error=str(e))
        
        # Method 3: Try direct column access
        if not row_dict:
            for col in tags_table.columns:
                try:
                    # Try attribute access first
                    if hasattr(row, col.name):
                        value = getattr(row, col.name, None)
                        if value is not None:
                            row_dict[col.name] = value
                    # Try index access
                    elif hasattr(row, '__getitem__'):
                        try:
                            # Try using the column object as key
                            value = row[col]
                            if value is not None:
                                row_dict[col.name] = value
                        except (KeyError, TypeError, IndexError):
                            pass
                except Exception as e:
                    logger.debug(f"Failed to get column {col.name}", error=str(e))
        
        # Method 4: Try keys() method (some Row implementations)
        if not row_dict and hasattr(row, 'keys'):
            try:
                row_dict = {key: row[key] for key in row.keys()}
            except Exception as e:
                logger.debug(f"Failed to convert using keys()", error=str(e))
        
        # Convert values to JSON-serializable format
        for col_name, value in row_dict.items():
            try:
                if value is None:
                    result[col_name] = None
                elif isinstance(value, (datetime, date)):
                    result[col_name] = value.isoformat()
                elif isinstance(value, bytes):
                    # Skip bytes - can't serialize
                    result[col_name] = None
                elif col_name == 'embedding':
                    # Skip embedding vector - can't serialize directly
                    result[col_name] = None
                elif isinstance(value, (dict, list)):
                    # Ensure nested structures are JSON-serializable
                    try:
                        json.dumps(value)
                        result[col_name] = value
                    except (TypeError, ValueError):
                        result[col_name] = self._clean_for_json(value)
                elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool)):
                    # Skip complex objects (but allow primitives)
                    result[col_name] = None
                else:
                    # Primitive types (str, int, float, bool) or simple iterables
                    result[col_name] = value
            except Exception as e:
                logger.debug(f"Failed to serialize column {col_name}", error=str(e))
                result[col_name] = None
        
        return result
    
    def _clean_for_json(self, obj):
        """Recursively clean object for JSON serialization"""
        import json
        
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    async def get_tags(
        self,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "name",
        sort_order: str = "asc"
    ) -> List[Dict[str, Any]]:
        """Get tags with search and filter support"""
        try:
            query = sa.select(tags_table)
            
            # Apply search filter
            if search:
                search_lower = search.lower()
                query = query.where(
                    or_(
                        func.lower(tags_table.c.name).contains(search_lower),
                        func.lower(tags_table.c.description).contains(search_lower) if tags_table.c.description else sa.false()
                    )
                )
            
            # Apply active filter
            if is_active is not None:
                query = query.where(tags_table.c.is_active == is_active)
            
            # Apply sorting
            if sort_by == "name":
                order_col = tags_table.c.name
            elif sort_by == "created_at":
                order_col = tags_table.c.created_at
            elif sort_by == "updated_at":
                order_col = tags_table.c.updated_at
            else:
                order_col = tags_table.c.name
            
            if sort_order == "desc":
                query = query.order_by(order_col.desc())
            else:
                query = query.order_by(order_col.asc())
            
            # Apply limit and offset
            query = query.limit(limit).offset(offset)
            
            rows = await self._execute_query(query, fetch="all")
            
            # Get blacklist status for each tag
            blacklisted_names = set()
            try:
                blacklist_query = sa.select(blacklisted_tags_table.c.tag_name)
                blacklist_rows = await self._execute_query(blacklist_query, fetch="all")
                for row in blacklist_rows or []:
                    if hasattr(row, '_mapping'):
                        tag_name = dict(row._mapping).get('tag_name')
                    elif hasattr(row, '_asdict'):
                        tag_name = row._asdict().get('tag_name')
                    elif hasattr(row, '__getitem__'):
                        tag_name = row[0] if len(row) > 0 else None
                    elif hasattr(row, 'tag_name'):
                        tag_name = row.tag_name
                    else:
                        tag_name = None
                    
                    if tag_name:
                        blacklisted_names.add(tag_name.lower())
            except Exception as e:
                logger.warning("Failed to get blacklist status", error=str(e))
            
            result = []
            for row in rows or []:
                try:
                    tag_dict = self._row_to_dict(row)
                    # Skip if we didn't get a valid tag dict or if name is missing
                    if not tag_dict or 'tag_id' not in tag_dict:
                        continue
                    
                    # Ensure name exists and is a string
                    tag_name = tag_dict.get('name')
                    if not tag_name or not isinstance(tag_name, str):
                        logger.debug("Skipping tag with invalid name", tag_id=tag_dict.get('tag_id'))
                        continue
                    
                    tag_dict['is_blacklisted'] = tag_name.lower() in blacklisted_names
                    
                    # Ensure prompts is a list and JSON-serializable
                    prompts = tag_dict.get('prompts')
                    if prompts is None:
                        tag_dict['prompts'] = []
                    elif not isinstance(prompts, list):
                        # Try to convert to list if it's a string or other type
                        try:
                            import json
                            if isinstance(prompts, str):
                                tag_dict['prompts'] = json.loads(prompts)
                            else:
                                tag_dict['prompts'] = list(prompts) if hasattr(prompts, '__iter__') and not isinstance(prompts, (str, bytes)) else []
                        except (json.JSONDecodeError, TypeError, ValueError):
                            tag_dict['prompts'] = []
                    
                    # Ensure all nested JSONB fields are properly formatted
                    for jsonb_field in ['parent_tag_ids', 'child_tag_ids', 'synonyms']:
                        if jsonb_field in tag_dict and tag_dict[jsonb_field] is not None:
                            if not isinstance(tag_dict[jsonb_field], list):
                                try:
                                    if isinstance(tag_dict[jsonb_field], str):
                                        import json
                                        tag_dict[jsonb_field] = json.loads(tag_dict[jsonb_field])
                                    else:
                                        tag_dict[jsonb_field] = []
                                except (json.JSONDecodeError, TypeError):
                                    tag_dict[jsonb_field] = []
                    
                    result.append(tag_dict)
                except Exception as e:
                    logger.warning("Failed to convert tag row to dict", error=str(e), exc_info=True)
                    continue
            
            return result

        except Exception as e:
            logger.error("Failed to get tags", error=str(e))
            raise

    async def get_tag(self, tag_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific tag by ID"""
        try:
            query = sa.select(tags_table).where(tags_table.c.tag_id == tag_id)
            row = await self._execute_query(query, fetch="one")
            
            if not row:
                return None
            
            tag_dict = self._row_to_dict(row)
            
            # Check if blacklisted - only if name exists
            tag_name = tag_dict.get('name')
            if tag_name and isinstance(tag_name, str):
                blacklist_query = sa.select(blacklisted_tags_table).where(
                    func.lower(blacklisted_tags_table.c.tag_name) == tag_name.lower()
                )
                blacklist_row = await self._execute_query(blacklist_query, fetch="one")
                
                if blacklist_row:
                    if hasattr(blacklist_row, '_mapping'):
                        blacklist_dict = dict(blacklist_row._mapping)
                    elif hasattr(blacklist_row, '_asdict'):
                        blacklist_dict = blacklist_row._asdict()
                    else:
                        blacklist_dict = {
                            'id': getattr(blacklist_row, 'id', None),
                            'tag_id': getattr(blacklist_row, 'tag_id', None),
                            'tag_name': getattr(blacklist_row, 'tag_name', None),
                            'reason': getattr(blacklist_row, 'reason', None)
                        }
                    tag_dict['is_blacklisted'] = True
                    tag_dict['blacklist_reason'] = blacklist_dict.get('reason')
                    tag_dict['blacklist_id'] = blacklist_dict.get('id')
                else:
                    tag_dict['is_blacklisted'] = False
                    tag_dict['blacklist_reason'] = None
                    tag_dict['blacklist_id'] = None
            else:
                tag_dict['is_blacklisted'] = False
                tag_dict['blacklist_reason'] = None
                tag_dict['blacklist_id'] = None
            
            # Ensure prompts is a list and JSON-serializable
            prompts = tag_dict.get('prompts')
            if prompts is None:
                tag_dict['prompts'] = []
            elif not isinstance(prompts, list):
                # Try to convert to list if it's a string or other type
                try:
                    import json
                    if isinstance(prompts, str):
                        tag_dict['prompts'] = json.loads(prompts)
                    else:
                        tag_dict['prompts'] = list(prompts) if hasattr(prompts, '__iter__') and not isinstance(prompts, (str, bytes)) else []
                except (json.JSONDecodeError, TypeError, ValueError):
                    tag_dict['prompts'] = []
            
            # Ensure JSONB fields are lists
            for jsonb_field in ['parent_tag_ids', 'child_tag_ids', 'synonyms']:
                if jsonb_field in tag_dict and tag_dict[jsonb_field] is not None:
                    if not isinstance(tag_dict[jsonb_field], list):
                        try:
                            if isinstance(tag_dict[jsonb_field], str):
                                import json
                                tag_dict[jsonb_field] = json.loads(tag_dict[jsonb_field])
                            else:
                                tag_dict[jsonb_field] = []
                        except (json.JSONDecodeError, TypeError):
                            tag_dict[jsonb_field] = []
            
            # Add aliases field (synonyms is used internally but we expose as aliases)
            if 'synonyms' in tag_dict:
                tag_dict['aliases'] = tag_dict['synonyms']
            
            return tag_dict

        except Exception as e:
            logger.error("Failed to get tag", tag_id=tag_id, error=str(e))
            raise

    async def update_tag(
        self,
        tag_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        parent_tag_ids: Optional[List[str]] = None,
        child_tag_ids: Optional[List[str]] = None,
        review_threshold: Optional[float] = None,
        auto_threshold: Optional[float] = None,
        is_active: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Update a tag"""
        try:
            async with async_session_maker() as session:
                # Build update values
                update_values = {}
                if name is not None:
                    update_values['name'] = name
                if description is not None:
                    update_values['description'] = description
                if prompts is not None:
                    update_values['prompts'] = prompts
                if aliases is not None:
                    update_values['synonyms'] = aliases  # Store aliases in synonyms column
                if parent_tag_ids is not None:
                    update_values['parent_tag_ids'] = parent_tag_ids
                if child_tag_ids is not None:
                    update_values['child_tag_ids'] = child_tag_ids
                if review_threshold is not None:
                    update_values['review_threshold'] = review_threshold
                if auto_threshold is not None:
                    update_values['auto_threshold'] = auto_threshold
                if is_active is not None:
                    update_values['is_active'] = is_active
                
                if not update_values:
                    # No updates, just return the tag
                    return await self.get_tag(tag_id)
                
                # Update tag
                update_query = (
                    sa.update(tags_table)
                    .where(tags_table.c.tag_id == tag_id)
                    .values(**update_values)
                )
                await session.execute(update_query)
                await session.commit()
                
                # Return updated tag
                return await self.get_tag(tag_id)

        except Exception as e:
            logger.error("Failed to update tag", tag_id=tag_id, error=str(e))
            raise

    async def get_blacklisted_tags(self) -> List[Dict[str, Any]]:
        """Get all blacklisted tags"""
        try:
            query = sa.select(blacklisted_tags_table)
            rows = await self._execute_query(query, fetch="all")
            
            result = []
            column_names = [col.name for col in blacklisted_tags_table.columns]
            
            for row in rows or []:
                try:
                    row_dict = {}
                    for col_name in column_names:
                        try:
                            # Try to get value using different methods
                            value = None
                            
                            if hasattr(row, '_mapping'):
                                try:
                                    mapping = row._mapping
                                    if col_name in mapping:
                                        value = mapping[col_name]
                                except (TypeError, ValueError, KeyError):
                                    pass
                            
                            if value is None and hasattr(row, col_name):
                                try:
                                    value = getattr(row, col_name)
                                except AttributeError:
                                    pass
                            
                            if value is None and hasattr(row, '__getitem__'):
                                try:
                                    col_idx = column_names.index(col_name)
                                    value = row[col_idx]
                                except (ValueError, IndexError, TypeError):
                                    pass
                            
                            # Serialize value
                            if value is None:
                                row_dict[col_name] = None
                            elif isinstance(value, (datetime, date)):
                                row_dict[col_name] = value.isoformat()
                            elif isinstance(value, bytes):
                                row_dict[col_name] = None
                            else:
                                row_dict[col_name] = value
                                
                        except Exception:
                            row_dict[col_name] = None
                    
                    result.append(row_dict)
                except Exception as e:
                    logger.warning("Failed to convert blacklist row to dict", error=str(e))
                    continue
            
            return result

        except Exception as e:
            logger.error("Failed to get blacklisted tags", error=str(e))
            raise

    async def add_to_blacklist(self, tag_name: str, tag_id: Optional[str] = None, reason: Optional[str] = None) -> Dict[str, Any]:
        """Add a tag to the blacklist"""
        try:
            async with async_session_maker() as session:
                # Check if already blacklisted (case-insensitive)
                existing_query = sa.select(blacklisted_tags_table).where(
                    func.lower(blacklisted_tags_table.c.tag_name) == tag_name.lower()
                )
                existing = await session.execute(existing_query)
                existing_row = existing.first()
                
                if existing_row:
                    # Update existing blacklist entry
                    update_query = (
                        sa.update(blacklisted_tags_table)
                        .where(blacklisted_tags_table.c.id == (existing_row.id if hasattr(existing_row, 'id') else existing_row[0]))
                        .values(
                            tag_id=tag_id,
                            tag_name=tag_name,
                            reason=reason
                        )
                    )
                    await session.execute(update_query)
                else:
                    # Insert new blacklist entry
                    insert_query = sa.insert(blacklisted_tags_table).values(
                        tag_id=tag_id,
                        tag_name=tag_name,
                        reason=reason
                    )
                    await session.execute(insert_query)
                
                await session.commit()
                
                # Return the blacklist entry
                query = sa.select(blacklisted_tags_table).where(
                    func.lower(blacklisted_tags_table.c.tag_name) == tag_name.lower()
                )
                result = await session.execute(query)
                row = result.first()
                
                if row:
                    if hasattr(row, '_mapping'):
                        return dict(row._mapping)
                    elif hasattr(row, '_asdict'):
                        return row._asdict()
                    else:
                        return {
                            'id': row.id,
                            'tag_id': row.tag_id,
                            'tag_name': row.tag_name,
                            'reason': row.reason,
                            'created_at': row.created_at,
                            'updated_at': row.updated_at
                        }
                
                raise Exception("Failed to retrieve blacklist entry after insert")

        except Exception as e:
            logger.error("Failed to add to blacklist", tag_name=tag_name, error=str(e))
            raise

    async def remove_from_blacklist(self, tag_name: str) -> bool:
        """Remove a tag from the blacklist"""
        try:
            async with async_session_maker() as session:
                delete_query = sa.delete(blacklisted_tags_table).where(
                    func.lower(blacklisted_tags_table.c.tag_name) == tag_name.lower()
                )
                result = await session.execute(delete_query)
                await session.commit()
                
                return result.rowcount > 0

        except Exception as e:
            logger.error("Failed to remove from blacklist", tag_name=tag_name, error=str(e))
            raise

    async def is_blacklisted(self, tag_name: str) -> bool:
        """Check if a tag is blacklisted"""
        try:
            query = sa.select(blacklisted_tags_table).where(
                func.lower(blacklisted_tags_table.c.tag_name) == tag_name.lower()
            )
            row = await self._execute_query(query, fetch="one")
            return row is not None

        except Exception as e:
            logger.error("Failed to check blacklist status", tag_name=tag_name, error=str(e))
            raise


def get_tags_service() -> TagsService:
    """Dependency to get tags service instance"""
    return TagsService()

