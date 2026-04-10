-- Run against your Vectera Postgres DB after chunks exist (or after a large bulk load).
-- HNSW speeds up approximate nearest-neighbor search vs sequential scan on embedding.
--
-- From host (example):
--   psql "$DATABASE_URL" -f scripts/pgvector_hnsw.sql
--
-- For a table already in production with many rows, prefer CONCURRENTLY to avoid long locks:
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_hnsw_idx
--   ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks
USING hnsw (embedding vector_cosine_ops);
