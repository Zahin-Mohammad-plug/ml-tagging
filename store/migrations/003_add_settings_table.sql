-- Migration: Add settings table
-- This table stores application settings that can be configured via the UI

CREATE TABLE IF NOT EXISTS settings (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    key VARCHAR NOT NULL UNIQUE,
    value JSONB NOT NULL DEFAULT '{}',
    user_id VARCHAR,  -- For future multi-user support
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);
CREATE INDEX IF NOT EXISTS idx_settings_user_id ON settings(user_id);

-- Apply updated_at trigger
CREATE TRIGGER update_settings_updated_at BEFORE UPDATE ON settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

