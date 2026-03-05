-- Migration: Add prompts column to tags table
-- This column stores sentence prompts used for tag matching

-- Add prompts column to tags table
ALTER TABLE tags ADD COLUMN IF NOT EXISTS prompts JSONB DEFAULT '[]';

-- Create index for prompt searches (optional, for future use)
-- Note: GIN index on JSONB for efficient array searches
CREATE INDEX IF NOT EXISTS idx_tags_prompts ON tags USING GIN (prompts);


