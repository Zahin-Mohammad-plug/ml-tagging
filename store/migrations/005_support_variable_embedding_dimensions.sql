-- Migration: Update embedding dimensions to 768 for CLIP ViT-L/14
-- This migration updates the embeddings and tags tables to use 768 dimensions
-- matching the default CLIP ViT-L/14 model (laion/CLIP-ViT-L-14-laion2B-s32B-b82K)
-- Note: pgvector requires exact dimension match between column and inserted vectors

-- Drop the old vector index
DROP INDEX IF EXISTS idx_embeddings_vector;

-- Alter the embedding column to 768 dimensions (CLIP ViT-L/14)
ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(768);

-- Recreate the vector index with new dimension
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Also update tags table to match
ALTER TABLE tags ALTER COLUMN embedding TYPE vector(768);



