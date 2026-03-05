"""Service for managing ML suggestions and approvals"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import pytz
import structlog
import sqlalchemy as sa

from ..database import (
    async_session_maker,
    suggestions_table,
    audit_log_table,
    tags_table,
    scenes_table,
    frame_samples_table,
    blacklisted_tags_table,
)
from ..models import (
    SuggestionResponse,
    SuggestionStatus,
    EvidenceFrame,
    ConfidenceBreakdown,
    TagContext,
)
from ..config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class SuggestionService:
    """Service for managing ML-generated tag suggestions"""

    async def _execute_query(self, query, *, fetch: Optional[str] = None, commit: bool = False):
        """Execute a query using a dedicated async session."""

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
            except Exception as e:
                await session.rollback()
                # Log the full error for debugging
                logger.error("Query execution failed", error=str(e), error_type=type(e).__name__, exc_info=True)
                raise

    async def get_suggestions(
        self,
        status: Optional[SuggestionStatus] = None,
        scene_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        include_backup: bool = False,  # Filter out backup tags by default
        limit: int = 50,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> List[SuggestionResponse]:
        """Get suggestions with optional filtering and sorting."""

        query = sa.select(suggestions_table)

        if status:
            query = query.where(suggestions_table.c.status == status)

        if scene_id:
            query = query.where(suggestions_table.c.scene_id == scene_id)
        
        if min_confidence is not None:
            query = query.where(suggestions_table.c.confidence >= min_confidence)
        
        # Filter out backup tags if not including them
        # Use SQLAlchemy 2.0 syntax: is_(False) or is_(None) for nullable booleans
        if not include_backup:
            query = query.where(
                sa.or_(
                    suggestions_table.c.is_backup.is_(False),
                    suggestions_table.c.is_backup.is_(None)
                )
            )

        # Handle sorting
        if sort_by:
            sort_column = None
            if sort_by == "confidence":
                sort_column = suggestions_table.c.confidence
            elif sort_by == "date" or sort_by == "created_at":
                sort_column = suggestions_table.c.created_at
            elif sort_by == "scene" or sort_by == "scene_id":
                sort_column = suggestions_table.c.scene_id
            
            # Check if sort_column is not None (can't use SQLAlchemy column in boolean context)
            if sort_column is not None:
                if sort_order and sort_order.lower() == "asc":
                    query = query.order_by(sort_column.asc())
                else:
                    query = query.order_by(sort_column.desc())
            else:
                # Default sorting if invalid sort_by
                query = query.order_by(
                    suggestions_table.c.confidence.desc(),
                    suggestions_table.c.created_at.desc(),
                )
        else:
            # Default sorting: highest confidence first
            query = query.order_by(
                suggestions_table.c.confidence.desc(),
                suggestions_table.c.created_at.desc(),
            )

        query = query.limit(limit).offset(offset)

        suggestions = await self._execute_query(query, fetch="all") or []

        # Get blacklisted tag names (case-insensitive)
        blacklist_query = sa.select(blacklisted_tags_table.c.tag_name)
        blacklist_rows = await self._execute_query(blacklist_query, fetch="all")
        blacklisted_names = set()
        for row in blacklist_rows or []:
            # Rows are already dictionaries from _execute_query
            if isinstance(row, dict):
                tag_name = row.get('tag_name')
            elif hasattr(row, '_mapping'):
                tag_name = dict(row._mapping).get('tag_name')
            elif hasattr(row, '_asdict'):
                tag_name = row._asdict().get('tag_name')
            elif hasattr(row, 'tag_name'):
                tag_name = row.tag_name
            else:
                # Fallback: try to get first value if it's a tuple-like object
                try:
                    tag_name = row[0] if len(row) > 0 else None
                except (TypeError, IndexError, KeyError):
                    tag_name = None
            if tag_name:
                blacklisted_names.add(tag_name.lower())

        result: List[SuggestionResponse] = []
        for suggestion in suggestions:
            # Suggestions are already dictionaries from _execute_query
            # No need to convert again
            suggestion_dict = suggestion if isinstance(suggestion, dict) else dict(suggestion)
            
            # Filter out blacklisted tags
            tag_name = suggestion_dict.get('tag_name', '').lower()
            if tag_name in blacklisted_names:
                continue  # Skip blacklisted tags
            
            enriched = await self._enrich_suggestion(suggestion_dict)
            result.append(enriched)

        return result

    async def get_suggestion(self, suggestion_id: str) -> Optional[SuggestionResponse]:
        """Get a specific suggestion with full details."""

        query = sa.select(suggestions_table).where(suggestions_table.c.id == suggestion_id)
        suggestion = await self._execute_query(query, fetch="one")

        if not suggestion:
            return None

        return await self._enrich_suggestion(dict(suggestion))

    async def approve_suggestion(
        self,
        suggestion_id: str,
        approved_by: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Approve a suggestion and update status."""

        update_data = {
            "status": SuggestionStatus.APPROVED,
            "reviewed_at": datetime.now(pytz.UTC),
            "reviewed_by": approved_by,
            "review_notes": notes,
        }

        query = sa.update(suggestions_table).where(suggestions_table.c.id == suggestion_id).values(**update_data)
        await self._execute_query(query, commit=True)

        await self._log_action(suggestion_id, "approved", approved_by, notes)
        logger.info("Suggestion approved", suggestion_id=suggestion_id, approved_by=approved_by)

    async def reject_suggestion(
        self,
        suggestion_id: str,
        rejected_by: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Reject a suggestion and update status."""

        update_data = {
            "status": SuggestionStatus.REJECTED,
            "reviewed_at": datetime.now(pytz.UTC),
            "reviewed_by": rejected_by,
            "review_notes": notes,
        }

        query = sa.update(suggestions_table).where(suggestions_table.c.id == suggestion_id).values(**update_data)
        await self._execute_query(query, commit=True)

        await self._log_action(suggestion_id, "rejected", rejected_by, notes)
        logger.info("Suggestion rejected", suggestion_id=suggestion_id, rejected_by=rejected_by)

    async def delete_suggestions(self, suggestion_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple suggestions by IDs"""
        try:
            async with async_session_maker() as session:
                # Delete suggestions
                delete_query = sa.delete(suggestions_table).where(
                    suggestions_table.c.id.in_(suggestion_ids)
                )
                result = await session.execute(delete_query)
                await session.commit()
                
                deleted_count = result.rowcount
                logger.info(
                    "Suggestions deleted",
                    count=deleted_count,
                    suggestion_ids=suggestion_ids
                )
                
                return {
                    "deleted_count": deleted_count,
                    "suggestion_ids": suggestion_ids
                }
        except Exception as e:
            logger.error("Failed to delete suggestions", error=str(e), exc_info=True)
            raise

    async def auto_apply_suggestion(self, suggestion_id: str, model_version: str):
        """Mark suggestion as auto-applied."""

        update_data = {
            "status": SuggestionStatus.AUTO_APPLIED,
            "reviewed_at": datetime.now(pytz.UTC),
            "reviewed_by": "system",
            "review_notes": f"Auto-applied by model {model_version}",
        }

        query = sa.update(suggestions_table).where(suggestions_table.c.id == suggestion_id).values(**update_data)
        await self._execute_query(query, commit=True)

        await self._log_action(
            suggestion_id,
            "auto_applied",
            "system",
            f"Auto-applied by model {model_version}",
            {"model_version": model_version},
        )

        logger.info("Suggestion auto-applied", suggestion_id=suggestion_id, model_version=model_version)

    async def get_stats(self) -> Dict[str, Any]:
        """Get suggestion statistics."""

        status_counts: Dict[str, int] = {}
        for status in SuggestionStatus:
            count_query = sa.select(sa.func.count()).select_from(suggestions_table).where(
                suggestions_table.c.status == status
            )
            count = await self._execute_query(count_query, fetch="scalar") or 0
            status_counts[status.value] = int(count)

        avg_confidence_query = sa.select(sa.func.avg(suggestions_table.c.confidence)).where(
            suggestions_table.c.status == SuggestionStatus.PENDING
        )
        avg_confidence = await self._execute_query(avg_confidence_query, fetch="scalar")

        approval_rate = None
        reviewed_count = status_counts.get("approved", 0) + status_counts.get("rejected", 0)
        if reviewed_count > 0:
            approval_rate = status_counts.get("approved", 0) / reviewed_count

        return {
            "total_suggestions": sum(status_counts.values()),
            "pending_count": status_counts.get("pending", 0),
            "approved_count": status_counts.get("approved", 0),
            "rejected_count": status_counts.get("rejected", 0),
            "auto_applied_count": status_counts.get("auto_applied", 0),
            "average_confidence": avg_confidence,
            "accuracy_rate": approval_rate,
        }

    async def _enrich_suggestion(self, suggestion: Dict[str, Any]) -> SuggestionResponse:
        """Enrich suggestion with additional data."""

        scene_query = sa.select(scenes_table).where(scenes_table.c.scene_id == suggestion["scene_id"])
        scene = await self._execute_query(scene_query, fetch="one")
        scene_title = scene.get("title") if scene else None

        tag_context = await self._get_tag_context(suggestion["tag_id"])

        confidence_breakdown = ConfidenceBreakdown(
            vision_confidence=suggestion.get("vision_confidence", 0.0),
            asr_confidence=suggestion.get("asr_confidence"),
            ocr_confidence=suggestion.get("ocr_confidence"),
            temporal_consistency=suggestion.get("temporal_consistency", 0.0),
            calibrated_confidence=suggestion.get("calibrated_confidence", suggestion["confidence"]),
        )

        # Parse evidence_frames - it might be a JSON string or already parsed
        evidence_frames_raw = suggestion.get("evidence_frames", [])
        if isinstance(evidence_frames_raw, str):
            import json
            try:
                evidence_frames_raw = json.loads(evidence_frames_raw)
            except:
                evidence_frames_raw = []
        
        # Handle evidence_frames - it might be an array of frame IDs (strings) or frame data objects
        evidence_frames = []
        for frame_data in evidence_frames_raw:
            if isinstance(frame_data, str):
                # Just a frame ID string - look up frame details from database
                frame_id = frame_data
                # Explicitly select the columns we need
                frame_query = sa.select(
                    frame_samples_table.c.id,
                    frame_samples_table.c.frame_number,
                    frame_samples_table.c.timestamp_seconds,
                    frame_samples_table.c.file_path
                ).where(frame_samples_table.c.id == frame_id)
                frame_record = await self._execute_query(frame_query, fetch="one")
                
                if frame_record:
                    # frame_record is already a dict from _execute_query
                    frame_dict = frame_record if isinstance(frame_record, dict) else dict(frame_record._mapping) if hasattr(frame_record, '_mapping') else {}
                    
                    # Create URL for frame image (always create URL, even if file_path is None)
                    # The endpoint will handle missing files gracefully
                    thumbnail_url = f"/api/frames/{frame_id}/image"
                    
                    # Handle None values for timestamp_seconds - ensure we get the actual value
                    timestamp = frame_dict.get("timestamp_seconds")
                    if timestamp is None or timestamp == "":
                        timestamp = 0.0
                    else:
                        try:
                            timestamp = float(timestamp)
                        except (ValueError, TypeError):
                            timestamp = 0.0
                    
                    # Handle None values for frame_number - ensure we get the actual value
                    frame_number = frame_dict.get("frame_number")
                    if frame_number is None or frame_number == "":
                        frame_number = 0
                    else:
                        try:
                            frame_number = int(frame_number)
                        except (ValueError, TypeError):
                            frame_number = 0
                    
                    # Use overall suggestion confidence as fallback for frame confidence
                    # Since we don't store individual frame confidence scores
                    frame_confidence = suggestion.get("confidence", 0.0)
                    
                    evidence_frames.append(
                        EvidenceFrame(
                            frame_number=frame_number,
                            timestamp_seconds=timestamp,
                            confidence=frame_confidence,  # Use suggestion confidence as fallback
                            thumbnail_url=thumbnail_url,
                            signals={"frame_id": frame_id, "file_path": frame_dict.get("file_path")},
                        )
                    )
                else:
                    # Frame not found, create minimal EvidenceFrame
                    evidence_frames.append(
                        EvidenceFrame(
                            frame_number=0,
                            timestamp_seconds=0.0,
                            confidence=0.0,
                            thumbnail_url=None,
                            signals={"frame_id": frame_id},
                        )
                    )
            elif isinstance(frame_data, dict):
                # Full frame data object
                frame_id = frame_data.get("frame_id") or frame_data.get("id")
                frame_file_path = frame_data.get("file_path")
                thumbnail_url = frame_data.get("thumbnail_url")
                
                # If we have frame_id but no thumbnail_url, create one
                if frame_id and not thumbnail_url and frame_file_path:
                    thumbnail_url = f"/api/frames/{frame_id}/image"
                
                # Handle None values for all fields
                frame_number = frame_data.get("frame_number")
                if frame_number is None:
                    frame_number = 0
                else:
                    frame_number = int(frame_number)
                
                timestamp = frame_data.get("timestamp_seconds")
                if timestamp is None:
                    timestamp = 0.0
                else:
                    timestamp = float(timestamp)
                
                confidence = frame_data.get("confidence")
                if confidence is None:
                    confidence = 0.0
                else:
                    confidence = float(confidence)
                
                evidence_frames.append(
                    EvidenceFrame(
                        frame_number=frame_number,
                        timestamp_seconds=timestamp,
                        confidence=confidence,
                        thumbnail_url=thumbnail_url,
                        signals=frame_data.get("signals", {}),
                    )
                )

        return SuggestionResponse(
            id=suggestion["id"],
            scene_id=suggestion["scene_id"],
            scene_title=scene_title,
            tag_context=tag_context,
            confidence=suggestion["confidence"],
            confidence_breakdown=confidence_breakdown,
            status=SuggestionStatus(suggestion["status"]),
            evidence_frames=evidence_frames,
            reasoning=suggestion.get("reasoning"),
            created_at=suggestion["created_at"],
            reviewed_at=suggestion.get("reviewed_at"),
            reviewed_by=suggestion.get("reviewed_by"),
            notes=suggestion.get("review_notes"),
            is_backup=suggestion.get("is_backup", False),
        )

    async def _get_tag_context(self, tag_id: str) -> TagContext:
        """Get tag context with hierarchy information."""

        query = sa.select(tags_table).where(tags_table.c.tag_id == tag_id)
        tag = await self._execute_query(query, fetch="one")

        if not tag:
            return TagContext(
                tag_id=tag_id,
                tag_name="Unknown Tag",
                parent_tags=[],
                child_tags=[],
                synonyms=[],
            )

        parent_names: List[str] = []
        parent_ids = tag.get("parent_tag_ids") or []
        if parent_ids:
            parent_query = sa.select(tags_table).where(tags_table.c.tag_id.in_(parent_ids))
            parents = await self._execute_query(parent_query, fetch="all") or []
            parent_names = [parent.get("name") for parent in parents]

        child_names: List[str] = []
        child_ids = tag.get("child_tag_ids") or []
        if child_ids:
            child_query = sa.select(tags_table).where(tags_table.c.tag_id.in_(child_ids))
            children = await self._execute_query(child_query, fetch="all") or []
            child_names = [child.get("name") for child in children]

        return TagContext(
            tag_id=tag.get("tag_id"),
            tag_name=tag.get("name", "Unknown Tag"),
            parent_tags=[name for name in parent_names if name],
            child_tags=[name for name in child_names if name],
            synonyms=tag.get("synonyms", []),
        )

    async def _log_action(
        self,
        suggestion_id: str,
        action: str,
        actor: Optional[str],
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log action in audit trail."""

        query = sa.select(suggestions_table).where(suggestions_table.c.id == suggestion_id)
        suggestion = await self._execute_query(query, fetch="one")
        confidence_at_action = suggestion.get("confidence") if suggestion else 0.0

        audit_data = {
            "suggestion_id": suggestion_id,
            "action": action,
            "actor": actor,
            "confidence_at_action": confidence_at_action,
            "notes": notes,
            "metadata": metadata or {},
        }

        insert_query = sa.insert(audit_log_table).values(**audit_data)
        await self._execute_query(insert_query, commit=True)

    async def get_suggestion_history(self, suggestion_id: str) -> List[Dict[str, Any]]:
        """Get audit history for a suggestion."""

        query = (
            sa.select(audit_log_table)
            .where(audit_log_table.c.suggestion_id == suggestion_id)
            .order_by(audit_log_table.c.created_at.asc())
        )

        history = await self._execute_query(query, fetch="all") or []
        return [dict(entry) for entry in history]