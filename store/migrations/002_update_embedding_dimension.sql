-- Migration: Update embedding dimension from 512 to 768 for CLIP ViT-L/14
-- This migration updates the embeddings table to support 768-dimensional vectors
-- 
-- NOTE: This migration will only run automatically on first database initialization.
-- If the database already exists, run this manually:
-- docker-compose exec postgres psql -U tagger -d tagger -f /docker-entrypoint-initdb.d/002_update_embedding_dimension.sql

-- Drop the old vector index
DROP INDEX IF EXISTS idx_embeddings_vector;

-- Alter the embedding column to support 768 dimensions
-- If table is empty, this will work. If it has data, you may need to clear it first.
ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(768);

-- Recreate the vector index with new dimension
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Also update tags table if it exists (tags table may be empty, so this should work)
ALTER TABLE tags ALTER COLUMN embedding TYPE vector(768);

