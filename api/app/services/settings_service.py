"""Service for managing application settings"""

from typing import Optional, Dict, Any, List
import structlog
import sqlalchemy as sa
from ..database import async_session_maker, settings_table
from ..config import get_settings, Settings

logger = structlog.get_logger()


class SettingsService:
    """Service for managing application settings"""

    def __init__(self):
        self.default_settings = get_settings()

    async def _execute_query(self, query, *, fetch: Optional[str] = None, commit: bool = False):
        """Execute a query using a dedicated async session."""
        async with async_session_maker() as session:
            try:
                result = await session.execute(query)
                
                # Materialize results BEFORE commit/close to ensure all data is loaded
                if fetch == "one":
                    row = result.first()
                    if row:
                        # Convert Row to dict immediately while session is open
                        if hasattr(row, '_mapping'):
                            row_dict = dict(row._mapping)
                        elif hasattr(row, '_asdict'):
                            row_dict = row._asdict()
                        else:
                            row_dict = {col.name: getattr(row, col.name, None) for col in result.column_descriptions}
                        if commit:
                            await session.commit()
                        return row_dict
                    else:
                        if commit:
                            await session.commit()
                        return None
                elif fetch == "all":
                    rows = result.fetchall()
                    # Convert Row objects to dicts immediately while session is open
                    rows_list = []
                    for row in rows:
                        if hasattr(row, '_mapping'):
                            rows_list.append(dict(row._mapping))
                        elif hasattr(row, '_asdict'):
                            rows_list.append(row._asdict())
                        else:
                            rows_list.append({col.name: getattr(row, col.name, None) for col in result.column_descriptions})
                    if commit:
                        await session.commit()
                    return rows_list
                elif fetch == "scalar":
                    scalar_value = result.scalar()
                    if commit:
                        await session.commit()
                    return scalar_value
                else:
                    if commit:
                        await session.commit()
                    return result
            except Exception:
                await session.rollback()
                raise

    def _get_default_value(self, key: str) -> Any:
        """Get default value for a setting key from config."""
        # Map setting keys to config attributes
        key_mapping = {
            "api_url": None,  # Not in config, UI-only
            "auto_approve": False,
            "confidence_threshold": self.default_settings.default_review_threshold,
            "sample_fps": self.default_settings.sample_fps,
            "max_frames_per_scene": self.default_settings.max_frames_per_scene,
            "min_agreement_frames": self.default_settings.min_agreement_frames,
            "temporal_window_seconds": self.default_settings.temporal_window_seconds,
            "vision_weight": self.default_settings.vision_weight,
            "asr_weight": self.default_settings.asr_weight,
            "ocr_weight": self.default_settings.ocr_weight,
            "default_review_threshold": self.default_settings.default_review_threshold,
            "default_auto_threshold": self.default_settings.default_auto_threshold,
            "vision_model": self.default_settings.vision_model,
            "vision_device": self.default_settings.vision_device,
            "asr_model": self.default_settings.asr_model,
            "asr_device": self.default_settings.asr_device,
            "ocr_engine": self.default_settings.ocr_engine,
            "clip_model_name": self.default_settings.clip_model_name,
            "use_calibrated_confidence": self.default_settings.use_calibrated_confidence,
            "calibration_model_path": self.default_settings.calibration_model_path,
            "cache_frames": self.default_settings.cache_frames,
            "max_concurrent_jobs": self.default_settings.max_concurrent_jobs,
            "job_timeout_seconds": self.default_settings.job_timeout_seconds,
        }
        return key_mapping.get(key)

    async def get_all_settings(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all settings, merging database values with defaults."""
        try:
            # Get all settings from database
            query = sa.select(settings_table)
            if user_id:
                query = query.where(settings_table.c.user_id == user_id)
            else:
                query = query.where(settings_table.c.user_id.is_(None))
            
            settings_rows = await self._execute_query(query, fetch="all")
            
            # Build settings dict from database
            # settings_rows is already a list of dicts from _execute_query
            db_settings = {}
            for row_dict in settings_rows or []:
                # row_dict is already a dict, just extract key and value
                if isinstance(row_dict, dict):
                    key = row_dict.get('key')
                    value = row_dict.get('value')
                    if key:
                        db_settings[key] = value
            
            # Merge with defaults
            result = {}
            # Add all default settings
            default_keys = [
                "auto_approve", "confidence_threshold",
                "sample_fps", "max_frames_per_scene", "min_agreement_frames",
                "temporal_window_seconds", "vision_weight", "asr_weight", "ocr_weight",
                "default_review_threshold", "default_auto_threshold",
                "vision_model", "vision_device", "asr_model", "asr_device",
                "ocr_engine", "clip_model_name", "use_calibrated_confidence", "calibration_model_path",
                "cache_frames", "max_concurrent_jobs", "job_timeout_seconds"
            ]
            for key in default_keys:
                default_value = self._get_default_value(key)
                result[key] = db_settings.get(key, default_value)
            
            # Add any additional keys from database
            for key, value in db_settings.items():
                if key not in result:
                    result[key] = value
            
            return result

        except Exception as e:
            logger.error("Failed to get settings", error=str(e))
            raise

    async def get_setting(self, key: str, user_id: Optional[str] = None) -> Any:
        """Get a specific setting value."""
        try:
            query = sa.select(settings_table).where(settings_table.c.key == key)
            if user_id:
                query = query.where(settings_table.c.user_id == user_id)
            else:
                query = query.where(settings_table.c.user_id.is_(None))
            
            row = await self._execute_query(query, fetch="one")
            
            if row:
                # row is already a dict from _execute_query
                if isinstance(row, dict):
                    return row.get('value')
                # Fallback for any edge cases
                return getattr(row, 'value', None) if hasattr(row, 'value') else None
            
            # Return default if not found
            return self._get_default_value(key)

        except Exception as e:
            logger.error("Failed to get setting", key=key, error=str(e))
            raise

    async def set_setting(self, key: str, value: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Set a setting value."""
        try:
            async with async_session_maker() as session:
                # Check if setting exists
                query = sa.select(settings_table).where(settings_table.c.key == key)
                if user_id:
                    query = query.where(settings_table.c.user_id == user_id)
                else:
                    query = query.where(settings_table.c.user_id.is_(None))
                
                result = await session.execute(query)
                existing_row = result.first()
                # Materialize the row immediately while session is open
                existing = dict(existing_row._mapping) if existing_row and hasattr(existing_row, '_mapping') else None
                
                import json
                # Convert value to JSONB-compatible format
                if isinstance(value, (dict, list)):
                    json_value = value
                else:
                    json_value = value
                
                if existing:
                    # Update existing setting
                    update_query = (
                        sa.update(settings_table)
                        .where(settings_table.c.key == key)
                        .values(value=json_value)
                    )
                    if user_id:
                        update_query = update_query.where(settings_table.c.user_id == user_id)
                    else:
                        update_query = update_query.where(settings_table.c.user_id.is_(None))
                    
                    await session.execute(update_query)
                else:
                    # Insert new setting
                    insert_query = sa.insert(settings_table).values(
                        key=key,
                        value=json_value,
                        user_id=user_id
                    )
                    await session.execute(insert_query)
                
                await session.commit()
                
                return {"key": key, "value": json_value}

        except Exception as e:
            logger.error("Failed to set setting", key=key, error=str(e))
            raise

    async def set_settings(self, settings: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Bulk update settings."""
        try:
            async with async_session_maker() as session:
                for key, value in settings.items():
                    # Check if setting exists
                    query = sa.select(settings_table).where(settings_table.c.key == key)
                    if user_id:
                        query = query.where(settings_table.c.user_id == user_id)
                    else:
                        query = query.where(settings_table.c.user_id.is_(None))
                    
                    result = await session.execute(query)
                    existing = result.first()
                    
                    # Convert value to JSONB-compatible format
                    if isinstance(value, (dict, list)):
                        json_value = value
                    else:
                        json_value = value
                    
                    if existing:
                        # Update existing setting
                        update_query = (
                            sa.update(settings_table)
                            .where(settings_table.c.key == key)
                            .values(value=json_value)
                        )
                        if user_id:
                            update_query = update_query.where(settings_table.c.user_id == user_id)
                        else:
                            update_query = update_query.where(settings_table.c.user_id.is_(None))
                        
                        await session.execute(update_query)
                    else:
                        # Insert new setting
                        insert_query = sa.insert(settings_table).values(
                            key=key,
                            value=json_value,
                            user_id=user_id
                        )
                        await session.execute(insert_query)
                
                await session.commit()
                
                return await self.get_all_settings(user_id=user_id)

        except Exception as e:
            logger.error("Failed to set settings", error=str(e))
            raise


def get_settings_service() -> SettingsService:
    """Dependency to get settings service instance"""
    return SettingsService()


