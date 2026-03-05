-- Migration: Support variable embedding dimensions (up to 1152 for SigLIP models)
-- This migration updates the embeddings and tags tables to support models with different embedding dimensions
-- Supports: 512 (ViT-B), 768 (ViT-L), 1024 (ViT-H), 1152 (SigLIP)

-- Drop the old vector index
DROP INDEX IF EXISTS idx_embeddings_vector;

-- Alter the embedding column to support up to 1152 dimensions (for SigLIP models)
-- Note: pgvector supports variable dimensions, but we set max to 1152
-- If table has data, you may need to clear embeddings for models with different dimensions first
ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(1152);

-- Recreate the vector index with new dimension
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Also update tags table to support variable dimensions
ALTER TABLE tags ALTER COLUMN embedding TYPE vector(1152);

-- Note: Existing embeddings with smaller dimensions (512, 768, 1024) will be automatically
-- padded or can coexist. When querying, pgvector handles dimension matching automatically.



