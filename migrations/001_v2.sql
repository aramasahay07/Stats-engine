-- Stats Engine v2 supplemental tables (keep your existing ones if already created)

-- jobs for progress + SSE
create table if not exists jobs (
  id uuid primary key,
  user_id text not null,
  dataset_id uuid not null,
  type text not null,
  status text not null,
  progress int not null default 0,
  message text,
  result_json jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- pipelines (PowerQuery-like transform steps)
create table if not exists pipelines (
  id uuid primary key,
  dataset_id uuid not null,
  user_id text not null,
  name text not null,
  steps_json jsonb not null,
  pipeline_hash text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_pipelines_dataset_user on pipelines(dataset_id, user_id);

-- Optional: projects (if not already)
create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  name text not null,
  created_at timestamptz not null default now()
);
