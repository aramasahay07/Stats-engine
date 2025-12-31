-- Add dataset lifecycle state + version for safe gating / reprocessing
-- 003_add_dataset_state.sql

ALTER TABLE datasets
  ADD COLUMN IF NOT EXISTS state TEXT NOT NULL DEFAULT 'ready';

ALTER TABLE datasets
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

ALTER TABLE datasets
  ADD COLUMN IF NOT EXISTS error_message TEXT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'datasets_state_check'
  ) THEN
    ALTER TABLE datasets
      ADD CONSTRAINT datasets_state_check
      CHECK (state IN ('ready','processing','reprocessing','failed'));
  END IF;
END $$;
