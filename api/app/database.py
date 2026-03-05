"""Database connection and initialization"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import Column, String, Float, DateTime, Boolean, Integer, Text, ForeignKey
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

from .config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()

# Create async engine with connection pool settings
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_size=10,  # Number of connections to maintain
    max_overflow=20,  # Maximum number of connections beyond pool_size
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,  # Don't autoflush to avoid unexpected commits
    autocommit=False,
)

# SQLAlchemy metadata
metadata = sa.MetaData()
Base = declarative_base()

# Database tables
scenes_table = sa.Table(
    "scenes",
    metadata,
    sa.Column("scene_id", sa.String, primary_key=True, index=True),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("path", sa.String, nullable=False),
    sa.Column("duration_seconds", sa.Float, nullable=True),
    sa.Column("frame_rate", sa.Float, nullable=True),
    sa.Column("width", sa.Integer, nullable=True),
    sa.Column("height", sa.Integer, nullable=True),
    sa.Column("last_processed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)

jobs_table = sa.Table(
    "jobs",
    metadata,
    sa.Column("job_id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("scene_id", sa.String, sa.ForeignKey("scenes.scene_id"), nullable=False, index=True),
    sa.Column("status", sa.String, nullable=False, default="queued", index=True),
    sa.Column("priority", sa.String, nullable=False, default="normal", index=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now(), index=True),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("error_message", sa.Text, nullable=True),
    sa.Column("progress", JSONB, default={}),
    sa.Column("metadata", JSONB, default={}),
)

frame_samples_table = sa.Table(
    "frame_samples", 
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("job_id", sa.String, sa.ForeignKey("jobs.job_id"), nullable=False, index=True),
    sa.Column("scene_id", sa.String, sa.ForeignKey("scenes.scene_id"), nullable=False, index=True),
    sa.Column("frame_number", sa.Integer, nullable=False),
    sa.Column("timestamp_seconds", sa.Float, nullable=False),
    sa.Column("file_path", sa.String, nullable=True),  # Path to extracted frame
    sa.Column("width", sa.Integer, nullable=True),
    sa.Column("height", sa.Integer, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
    sa.Index("idx_frame_samples_scene_timestamp", "scene_id", "timestamp_seconds"),
)

# Embeddings table with pgvector support
embeddings_table = sa.Table(
    "embeddings",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("frame_id", sa.String, sa.ForeignKey("frame_samples.id"), nullable=False, index=True),
    sa.Column("model_name", sa.String, nullable=False, index=True),
    sa.Column("embedding", Vector(1152), nullable=False),  # Supports up to 1152 (SigLIP), also 768 (ViT-L), 1024 (ViT-H)
    sa.Column("metadata", JSONB, default={}),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
    sa.Index("idx_embeddings_model", "model_name"),
)

# Text analysis results (ASR/OCR)
text_analysis_table = sa.Table(
    "text_analysis",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("frame_id", sa.String, sa.ForeignKey("frame_samples.id"), nullable=True, index=True),
    sa.Column("job_id", sa.String, sa.ForeignKey("jobs.job_id"), nullable=False, index=True),
    sa.Column("analysis_type", sa.String, nullable=False, index=True),  # 'asr' or 'ocr'
    sa.Column("text_content", sa.Text, nullable=False),
    sa.Column("confidence", sa.Float, nullable=True),
    sa.Column("language", sa.String, nullable=True),
    sa.Column("start_time", sa.Float, nullable=True),  # For ASR chunks
    sa.Column("end_time", sa.Float, nullable=True),    # For ASR chunks
    sa.Column("bounding_box", JSONB, nullable=True),   # For OCR text locations
    sa.Column("metadata", JSONB, default={}),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
)

# Tag definitions and hierarchies
tags_table = sa.Table(
    "tags",
    metadata,
    sa.Column("tag_id", sa.String, primary_key=True),  # Tag ID
    sa.Column("name", sa.String, nullable=False, index=True),
    sa.Column("description", sa.Text, nullable=True),
    sa.Column("parent_tag_ids", JSONB, default=[]),    # Parent tag IDs
    sa.Column("child_tag_ids", JSONB, default=[]),     # Child tag IDs
    sa.Column("synonyms", JSONB, default=[]),          # Alternative names for matching
    sa.Column("embedding", Vector(1152), nullable=True), # Tag name embedding - supports variable dimensions (512-1152)
    sa.Column("prompts", JSONB, default=[]),           # Sentence prompts for tag matching
    sa.Column("review_threshold", sa.Float, nullable=True),  # Custom threshold for this tag
    sa.Column("auto_threshold", sa.Float, nullable=True),    # Custom auto-apply threshold
    sa.Column("is_active", sa.Boolean, default=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)

# ML suggestions
suggestions_table = sa.Table(
    "suggestions",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("job_id", sa.String, sa.ForeignKey("jobs.job_id"), nullable=False, index=True),
    sa.Column("scene_id", sa.String, sa.ForeignKey("scenes.scene_id"), nullable=False, index=True),
    sa.Column("tag_id", sa.String, sa.ForeignKey("tags.tag_id", onupdate="CASCADE"), nullable=False, index=True),
    sa.Column("tag_name", sa.String, nullable=False, index=True),
    
    # Confidence scores
    sa.Column("confidence", sa.Float, nullable=False, index=True),
    sa.Column("vision_confidence", sa.Float, nullable=True),
    sa.Column("asr_confidence", sa.Float, nullable=True), 
    sa.Column("ocr_confidence", sa.Float, nullable=True),
    sa.Column("temporal_consistency", sa.Float, nullable=True),
    sa.Column("calibrated_confidence", sa.Float, nullable=True),
    
    # Evidence and reasoning
    sa.Column("evidence_frames", JSONB, default=[]),   # Frame IDs and details
    sa.Column("reasoning", sa.Text, nullable=True),
    sa.Column("signal_details", JSONB, default={}),
    
    # Status and review
    sa.Column("status", sa.String, nullable=False, default="pending", index=True),
    sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("reviewed_by", sa.String, nullable=True),
    sa.Column("review_notes", sa.Text, nullable=True),
    
    # Hierarchical filtering
    sa.Column("is_backup", sa.Boolean, default=False, nullable=True),  # True if this is a backup (parent) tag
    
    # Timestamps
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now(), index=True),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
    
    # Constraints
    sa.Index("idx_suggestions_scene_tag", "scene_id", "tag_id"),
    sa.Index("idx_suggestions_confidence", "confidence"),
    sa.Index("idx_suggestions_status_confidence", "status", "confidence"),
    sa.Index("idx_suggestions_is_backup", "is_backup"),
)

# Audit log for approvals/rejections
audit_log_table = sa.Table(
    "audit_log",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("suggestion_id", sa.String, sa.ForeignKey("suggestions.id"), nullable=False, index=True),
    sa.Column("action", sa.String, nullable=False, index=True),  # 'approved', 'rejected', 'auto_applied'
    sa.Column("actor", sa.String, nullable=True),  # User identifier (nullable for anonymous)
    sa.Column("confidence_at_action", sa.Float, nullable=False),
    sa.Column("model_version", sa.String, nullable=True),
    sa.Column("notes", sa.Text, nullable=True),
    sa.Column("metadata", JSONB, default={}),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now(), index=True),
)

# Performance metrics
metrics_table = sa.Table(
    "metrics",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("metric_type", sa.String, nullable=False, index=True),  # 'job_duration', 'suggestion_accuracy', etc.
    sa.Column("metric_name", sa.String, nullable=False, index=True),
    sa.Column("value", sa.Float, nullable=False),
    sa.Column("tags", JSONB, default={}),  # Additional dimensions
    sa.Column("timestamp", sa.DateTime(timezone=True), server_default=func.now(), index=True),
)

# Settings table (application settings)
settings_table = sa.Table(
    "settings",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("key", sa.String, nullable=False, unique=True, index=True),
    sa.Column("value", JSONB, nullable=False, default={}),
    sa.Column("user_id", sa.String, nullable=True, index=True),  # For future multi-user support
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)

# Blacklisted tags table
blacklisted_tags_table = sa.Table(
    "blacklisted_tags",
    metadata,
    sa.Column("id", sa.String, primary_key=True, default=lambda: str(uuid.uuid4())),
    sa.Column("tag_id", sa.String, sa.ForeignKey("tags.tag_id", ondelete="SET NULL", onupdate="CASCADE"), nullable=True),
    sa.Column("tag_name", sa.String, nullable=False, index=True),
    sa.Column("reason", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=func.now()),
    sa.Column("updated_at", sa.DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)

async def init_db():
    """Initialize database tables"""
    # Create tables using async engine
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        
        # Enable pgvector extension
        try:
            await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception as e:
            logger.warning("Vector extension issue (may already exist)", error=str(e))
    
    # Run migrations in separate transactions to avoid aborting on errors
    # Each operation is wrapped in its own transaction block
    
    # Settings table
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS settings (
                    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    key VARCHAR NOT NULL UNIQUE,
                    value JSONB NOT NULL DEFAULT '{}',
                    user_id VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))
    except Exception as e:
        logger.warning("Settings table creation issue", error=str(e))
    
    # Settings indexes
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);"))
            await conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_settings_user_id ON settings(user_id);"))
    except Exception as e:
        logger.warning("Settings indexes creation issue", error=str(e))
    
    # Blacklisted tags table
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS blacklisted_tags (
                    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
                    tag_id VARCHAR REFERENCES tags(tag_id) ON DELETE SET NULL,
                    tag_name VARCHAR NOT NULL,
                    reason TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))
    except Exception as e:
        logger.warning("Blacklisted_tags table creation issue", error=str(e))
    
    # Blacklisted tags indexes
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_blacklisted_tags_tag_id ON blacklisted_tags(tag_id);"))
            await conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_blacklisted_tags_tag_name ON blacklisted_tags(tag_name);"))
    except Exception as e:
        logger.warning("Blacklisted_tags indexes creation issue", error=str(e))
    
    # Unique index for blacklisted_tags (case-insensitive)
    # Note: PostgreSQL doesn't support expressions in UNIQUE constraints,
    # so we use a unique index instead
    try:
        async with engine.begin() as conn:
            # First, check if there's an invalid constraint with this name and drop it
            constraint_check = await conn.execute(sa.text("""
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'unique_tag_name_case_insensitive'
            """))
            if constraint_check.fetchone():
                # Drop the invalid constraint if it exists
                await conn.execute(sa.text("""
                    ALTER TABLE blacklisted_tags 
                    DROP CONSTRAINT IF EXISTS unique_tag_name_case_insensitive
                """))
                logger.info("Dropped invalid constraint unique_tag_name_case_insensitive")
            
            # Check if index exists
            index_check = await conn.execute(sa.text("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = 'unique_tag_name_case_insensitive'
            """))
            if not index_check.fetchone():
                await conn.execute(sa.text("""
                    CREATE UNIQUE INDEX unique_tag_name_case_insensitive 
                    ON blacklisted_tags (LOWER(tag_name));
                """))
                logger.info("Created case-insensitive unique index on blacklisted_tags.tag_name")
    except Exception as e:
        logger.warning("Blacklisted_tags unique index creation issue", error=str(e))
    
    # Add prompts column to tags table
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("ALTER TABLE tags ADD COLUMN IF NOT EXISTS prompts JSONB DEFAULT '[]';"))
    except Exception as e:
        logger.warning("Prompts column creation issue", error=str(e))
    
    # Tags prompts index
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_tags_prompts ON tags USING GIN (prompts);"))
    except Exception as e:
        logger.warning("Tags prompts index creation issue", error=str(e))
    
    # Add is_backup column to suggestions table
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("ALTER TABLE suggestions ADD COLUMN IF NOT EXISTS is_backup BOOLEAN DEFAULT FALSE;"))
    except Exception as e:
        logger.warning("is_backup column creation issue", error=str(e))
    
    # Add index for is_backup
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_suggestions_is_backup ON suggestions(is_backup);"))
    except Exception as e:
        logger.warning("is_backup index creation issue", error=str(e))
    
    # Create vector index (only if embeddings table has data)
    try:
        async with engine.begin() as conn:
            # Check if embeddings table exists and has data
            result = await conn.execute(sa.text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'embeddings'
            """))
            table_exists = result.scalar() > 0
            
            if table_exists:
                # Check if index already exists
                index_result = await conn.execute(sa.text("""
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = 'idx_embeddings_vector'
                """))
                if not index_result.fetchone():
                    # Check if table has any rows (ivfflat requires data)
                    count_result = await conn.execute(sa.text("SELECT COUNT(*) FROM embeddings"))
                    row_count = count_result.scalar()
                    if row_count > 0:
                        await conn.execute(sa.text("""
                            CREATE INDEX idx_embeddings_vector 
                            ON embeddings USING ivfflat (embedding vector_cosine_ops) 
                            WITH (lists = 100);
                        """))
                    else:
                        logger.info("Skipping vector index creation - embeddings table is empty")
    except Exception as e:
        logger.warning("Embeddings vector index creation skipped", error=str(e))
    
    # Create trigger function
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))
    except Exception as e:
        logger.warning("Trigger function creation issue", error=str(e))
    
    # Settings trigger
    try:
        async with engine.begin() as conn:
            # Check if trigger exists
            trigger_result = await conn.execute(sa.text("""
                SELECT 1 FROM pg_trigger WHERE tgname = 'update_settings_updated_at'
            """))
            if not trigger_result.fetchone():
                await conn.execute(sa.text("""
                    CREATE TRIGGER update_settings_updated_at 
                    BEFORE UPDATE ON settings
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                """))
    except Exception as e:
        logger.warning("Settings trigger creation issue", error=str(e))
    
    # Blacklisted tags trigger
    try:
        async with engine.begin() as conn:
            trigger_result = await conn.execute(sa.text("""
                SELECT 1 FROM pg_trigger WHERE tgname = 'update_blacklisted_tags_updated_at'
            """))
            if not trigger_result.fetchone():
                await conn.execute(sa.text("""
                    CREATE TRIGGER update_blacklisted_tags_updated_at 
                    BEFORE UPDATE ON blacklisted_tags
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                """))
    except Exception as e:
        logger.warning("Blacklisted_tags trigger creation issue", error=str(e))
    
    # Migration: Add ON UPDATE CASCADE to foreign key constraints
    try:
        async with engine.begin() as conn:
            # Check if suggestions_tag_id_fkey exists and update it
            constraint_check = await conn.execute(sa.text("""
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'suggestions_tag_id_fkey'
            """))
            if constraint_check.fetchone():
                # Drop and recreate with ON UPDATE CASCADE
                await conn.execute(sa.text("ALTER TABLE suggestions DROP CONSTRAINT IF EXISTS suggestions_tag_id_fkey"))
                await conn.execute(sa.text("""
                    ALTER TABLE suggestions 
                    ADD CONSTRAINT suggestions_tag_id_fkey 
                    FOREIGN KEY (tag_id) 
                    REFERENCES tags(tag_id) 
                    ON UPDATE CASCADE
                """))
                logger.info("Updated suggestions_tag_id_fkey constraint with ON UPDATE CASCADE")
    except Exception as e:
        logger.warning("Failed to update suggestions_tag_id_fkey constraint", error=str(e))
    
    # Migration: Add ON UPDATE CASCADE to blacklisted_tags foreign key
    try:
        async with engine.begin() as conn:
            constraint_check = await conn.execute(sa.text("""
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'blacklisted_tags_tag_id_fkey'
            """))
            if constraint_check.fetchone():
                # Check if it already has ON UPDATE CASCADE by examining the constraint definition
                constraint_def = await conn.execute(sa.text("""
                    SELECT pg_get_constraintdef(oid) as def
                    FROM pg_constraint 
                    WHERE conname = 'blacklisted_tags_tag_id_fkey'
                """))
                constraint_def_row = constraint_def.fetchone()
                if constraint_def_row and 'ON UPDATE CASCADE' not in constraint_def_row[0]:
                    # Drop and recreate with ON UPDATE CASCADE
                    await conn.execute(sa.text("ALTER TABLE blacklisted_tags DROP CONSTRAINT IF EXISTS blacklisted_tags_tag_id_fkey"))
                    await conn.execute(sa.text("""
                        ALTER TABLE blacklisted_tags 
                        ADD CONSTRAINT blacklisted_tags_tag_id_fkey 
                        FOREIGN KEY (tag_id) 
                        REFERENCES tags(tag_id) 
                        ON DELETE SET NULL 
                        ON UPDATE CASCADE
                    """))
                    logger.info("Updated blacklisted_tags_tag_id_fkey constraint with ON UPDATE CASCADE")
    except Exception as e:
        logger.warning("Failed to update blacklisted_tags_tag_id_fkey constraint", error=str(e))

async def get_database():
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()