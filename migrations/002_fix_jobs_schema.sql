-- Fix "jobs" schema mismatches that cause POST /datasets to fail
--
-- Goal schema (matches app/services/jobs_service.py):
--   jobs(id uuid primary key,
--        user_id text not null,
--        dataset_id uuid not null,
--        type text not null,
--        status text not null,
--        progress int not null default 0,
--        message text,
--        result_json jsonb,
--        created_at timestamptz not null default now(),
--        updated_at timestamptz not null default now());

do $$
begin
  -- Rename legacy columns if present
  if exists (
    select 1 from information_schema.columns
    where table_name='jobs' and column_name='job_id'
  ) then
    alter table jobs rename column job_id to id;
  end if;

  if exists (
    select 1 from information_schema.columns
    where table_name='jobs' and column_name='job_type'
  ) then
    alter table jobs rename column job_type to type;
  end if;

  -- Ensure required columns exist
  if not exists (
    select 1 from information_schema.columns
    where table_name='jobs' and column_name='id'
  ) then
    alter table jobs add column id uuid;
    update jobs set id = gen_random_uuid() where id is null;
    alter table jobs alter column id set not null;
    alter table jobs add primary key (id);
  end if;

  if not exists (
    select 1 from information_schema.columns
    where table_name='jobs' and column_name='type'
  ) then
    alter table jobs add column type text;
    update jobs set type = coalesce(type, 'unknown') where type is null;
    alter table jobs alter column type set not null;
  end if;

  -- Make user_id text (fixes: invalid UUID 'madbush')
  begin
    alter table jobs alter column user_id type text using user_id::text;
  exception when others then
    -- ignore if already text or cast not needed
  end;

  -- Ensure timestamps exist
  if not exists (
    select 1 from information_schema.columns
    where table_name='jobs' and column_name='created_at'
  ) then
    alter table jobs add column created_at timestamptz not null default now();
  end if;
  if not exists (
    select 1 from information_schema.columns
    where table_name='jobs' and column_name='updated_at'
  ) then
    alter table jobs add column updated_at timestamptz not null default now();
  end if;
end $$;
