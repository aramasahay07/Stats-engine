-- Session to dataset persistent mapping
CREATE TABLE IF NOT EXISTS session_dataset_links (
    session_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    user_id TEXT,
    project_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_updated_at_session_dataset_links()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS session_dataset_links_set_updated_at ON session_dataset_links;
CREATE TRIGGER session_dataset_links_set_updated_at
BEFORE UPDATE ON session_dataset_links
FOR EACH ROW EXECUTE FUNCTION update_updated_at_session_dataset_links();
