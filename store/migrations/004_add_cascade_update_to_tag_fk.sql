-- Migration: Add ON UPDATE CASCADE to suggestions.tag_id foreign key constraint
-- This allows tag_id to be updated in the tags table, and all references in suggestions
-- will be automatically updated to maintain referential integrity.
--
-- NOTE: This migration will only run automatically on first database initialization.
-- If the database already exists, run this manually:
-- docker-compose exec postgres psql -U tagger -d tagger -f /docker-entrypoint-initdb.d/004_add_cascade_update_to_tag_fk.sql

-- Drop the existing foreign key constraint
DO $$ 
BEGIN
    -- Check if the constraint exists and drop it
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'suggestions_tag_id_fkey'
    ) THEN
        ALTER TABLE suggestions DROP CONSTRAINT suggestions_tag_id_fkey;
    END IF;
END $$;

-- Recreate the foreign key constraint with ON UPDATE CASCADE
ALTER TABLE suggestions 
ADD CONSTRAINT suggestions_tag_id_fkey 
FOREIGN KEY (tag_id) 
REFERENCES tags(tag_id) 
ON UPDATE CASCADE;

-- Also update the blacklisted_tags foreign key if it exists (it already has ON DELETE SET NULL)
-- Check if it needs ON UPDATE CASCADE as well
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'blacklisted_tags_tag_id_fkey'
        AND conname IS NOT NULL
    ) THEN
        -- Drop and recreate with ON UPDATE CASCADE
        ALTER TABLE blacklisted_tags DROP CONSTRAINT blacklisted_tags_tag_id_fkey;
        ALTER TABLE blacklisted_tags 
        ADD CONSTRAINT blacklisted_tags_tag_id_fkey 
        FOREIGN KEY (tag_id) 
        REFERENCES tags(tag_id) 
        ON DELETE SET NULL 
        ON UPDATE CASCADE;
    END IF;
END $$;

