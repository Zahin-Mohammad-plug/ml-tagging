-- Create the pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the main database schema for ML Tagger
-- This file initializes the database tables and indexes

-- Scenes table (cached scene metadata)
CREATE TABLE IF NOT EXISTS scenes (
    scene_id VARCHAR PRIMARY KEY,
    title VARCHAR,
    path VARCHAR NOT NULL,
    duration_seconds FLOAT,
    frame_rate FLOAT,
    width INTEGER,
    height INTEGER,
    last_processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Jobs table (processing jobs)
CREATE TABLE IF NOT EXISTS jobs (
    job_id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    scene_id VARCHAR NOT NULL REFERENCES scenes(scene_id),
    status VARCHAR NOT NULL DEFAULT 'queued',
    priority VARCHAR NOT NULL DEFAULT 'normal',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    progress JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Frame samples (extracted video frames)
CREATE TABLE IF NOT EXISTS frame_samples (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    job_id VARCHAR NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    scene_id VARCHAR NOT NULL REFERENCES scenes(scene_id),
    frame_number INTEGER NOT NULL,
    timestamp_seconds FLOAT NOT NULL,
    file_path VARCHAR,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings (vector representations of frames)
CREATE TABLE IF NOT EXISTS embeddings (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    frame_id VARCHAR NOT NULL REFERENCES frame_samples(id) ON DELETE CASCADE,
    model_name VARCHAR NOT NULL,
    embedding VECTOR(512) NOT NULL, -- Standard CLIP dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Text analysis (ASR and OCR results)
CREATE TABLE IF NOT EXISTS text_analysis (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    frame_id VARCHAR REFERENCES frame_samples(id) ON DELETE CASCADE,
    job_id VARCHAR NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    analysis_type VARCHAR NOT NULL, -- 'asr' or 'ocr'
    text_content TEXT NOT NULL,
    confidence FLOAT,
    language VARCHAR,
    start_time FLOAT, -- For ASR segments
    end_time FLOAT,   -- For ASR segments
    bounding_box JSONB, -- For OCR text locations
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tags (tag definitions and hierarchies)
CREATE TABLE IF NOT EXISTS tags (
    tag_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    parent_tag_ids JSONB DEFAULT '[]',
    child_tag_ids JSONB DEFAULT '[]',
    synonyms JSONB DEFAULT '[]',
    embedding VECTOR(512), -- Tag name embedding
    review_threshold FLOAT,
    auto_threshold FLOAT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Suggestions (ML-generated tag suggestions)
CREATE TABLE IF NOT EXISTS suggestions (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    job_id VARCHAR NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    scene_id VARCHAR NOT NULL REFERENCES scenes(scene_id),
    tag_id VARCHAR NOT NULL REFERENCES tags(tag_id),
    tag_name VARCHAR NOT NULL,
    
    -- Confidence scores
    confidence FLOAT NOT NULL,
    vision_confidence FLOAT,
    asr_confidence FLOAT,
    ocr_confidence FLOAT,
    temporal_consistency FLOAT,
    calibrated_confidence FLOAT,
    
    -- Evidence and reasoning
    evidence_frames JSONB DEFAULT '[]',
    reasoning TEXT,
    signal_details JSONB DEFAULT '{}',
    
    -- Status and review
    status VARCHAR NOT NULL DEFAULT 'pending',
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by VARCHAR,
    review_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log (approval/rejection history)
CREATE TABLE IF NOT EXISTS audit_log (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    suggestion_id VARCHAR NOT NULL REFERENCES suggestions(id) ON DELETE CASCADE,
    action VARCHAR NOT NULL, -- 'approved', 'rejected', 'auto_applied'
    actor VARCHAR, -- User identifier (nullable for anonymous)
    confidence_at_action FLOAT NOT NULL,
    model_version VARCHAR,
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Metrics (performance tracking)
CREATE TABLE IF NOT EXISTS metrics (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    metric_type VARCHAR NOT NULL, -- 'job_duration', 'suggestion_accuracy', etc.
    metric_name VARCHAR NOT NULL,
    value FLOAT NOT NULL,
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance

-- Job indexes
CREATE INDEX IF NOT EXISTS idx_jobs_scene_id ON jobs(scene_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

-- Frame sample indexes
CREATE INDEX IF NOT EXISTS idx_frame_samples_job_id ON frame_samples(job_id);
CREATE INDEX IF NOT EXISTS idx_frame_samples_scene_id ON frame_samples(scene_id);
CREATE INDEX IF NOT EXISTS idx_frame_samples_scene_timestamp ON frame_samples(scene_id, timestamp_seconds);

-- Embedding indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_frame_id ON embeddings(frame_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);

-- Vector similarity index (for fast similarity search)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Text analysis indexes
CREATE INDEX IF NOT EXISTS idx_text_analysis_frame_id ON text_analysis(frame_id);
CREATE INDEX IF NOT EXISTS idx_text_analysis_job_id ON text_analysis(job_id);
CREATE INDEX IF NOT EXISTS idx_text_analysis_type ON text_analysis(analysis_type);

-- Tag indexes
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_active ON tags(is_active);

-- Suggestion indexes
CREATE INDEX IF NOT EXISTS idx_suggestions_job_id ON suggestions(job_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_scene_id ON suggestions(scene_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_tag_id ON suggestions(tag_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_tag_name ON suggestions(tag_name);
CREATE INDEX IF NOT EXISTS idx_suggestions_confidence ON suggestions(confidence);
CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status);
CREATE INDEX IF NOT EXISTS idx_suggestions_status_confidence ON suggestions(status, confidence);
CREATE INDEX IF NOT EXISTS idx_suggestions_scene_tag ON suggestions(scene_id, tag_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_created_at ON suggestions(created_at);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_suggestion_id ON audit_log(suggestion_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Metrics indexes
CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

-- Create trigger for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_scenes_updated_at BEFORE UPDATE ON scenes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tags_updated_at BEFORE UPDATE ON tags
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_suggestions_updated_at BEFORE UPDATE ON suggestions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample tags for demo (cooking domain)
INSERT INTO tags (tag_id, name, synonyms, review_threshold, auto_threshold) VALUES 
('tag_001', 'Chopping', '["cutting", "dicing", "slicing", "mincing"]', 0.4, 0.8),
('tag_002', 'Boiling', '["simmering", "blanching", "poaching"]', 0.4, 0.8),
('tag_003', 'Grilling', '["barbecue", "BBQ", "charring"]', 0.4, 0.8),
('tag_004', 'Baking', '["oven", "roasting", "broiling"]', 0.3, 0.7),
('tag_005', 'Sauteing', '["pan frying", "stir frying"]', 0.3, 0.7),
('tag_006', 'Mixing', '["stirring", "whisking", "folding"]', 0.3, 0.7),
('tag_007', 'Plating', '["garnishing", "presentation"]', 0.4, 0.8),
('tag_008', 'Kneading', '["dough", "bread making"]', 0.4, 0.8),
('tag_009', 'Deep Frying', '["frying", "oil frying", "tempura"]', 0.4, 0.8),
('tag_010', 'Seasoning', '["spicing", "marinating", "salting"]', 0.4, 0.8)
ON CONFLICT (tag_id) DO NOTHING;

-- Create a view for suggestion analytics
CREATE OR REPLACE VIEW suggestion_analytics AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    status,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM suggestions
GROUP BY DATE_TRUNC('day', created_at), status
ORDER BY date DESC, status;