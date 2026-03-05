-- Migration: Add blacklisted_tags table
-- This table stores tags that should be automatically filtered out from suggestions

CREATE TABLE IF NOT EXISTS blacklisted_tags (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tag_id VARCHAR REFERENCES tags(tag_id) ON DELETE SET NULL,
    tag_name VARCHAR NOT NULL,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_blacklisted_tags_tag_id ON blacklisted_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_blacklisted_tags_tag_name ON blacklisted_tags(tag_name);

-- Create unique index for case-insensitive tag_name uniqueness
-- Note: PostgreSQL doesn't support expressions in UNIQUE constraints,
-- so we use a unique index instead
CREATE UNIQUE INDEX IF NOT EXISTS unique_tag_name_case_insensitive 
ON blacklisted_tags (LOWER(tag_name));

-- Apply updated_at trigger
CREATE TRIGGER update_blacklisted_tags_updated_at BEFORE UPDATE ON blacklisted_tags
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

