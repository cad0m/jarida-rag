import psycopg
from psycopg.rows import dict_row

from jarida_rag import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    url text PRIMARY KEY,
    module text NOT NULL,
    issue text,
    date date,
    filename text NOT NULL,
    path text NOT NULL,
    status text NOT NULL DEFAULT 'listed',
    attempts int NOT NULL DEFAULT 0,
    size bigint,
    sha256 text,
    error text,
    updated_at timestamptz NOT NULL DEFAULT now()
)"""
MAX_ATTEMPTS = 4


def connect():
    # not store.connect(): store loads torch, which listing and downloading don't need
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    db = psycopg.connect(config.DATABASE_URL, autocommit=True, row_factory=dict_row)
    db.execute(SCHEMA)
    return db


def save_listed(db, issues):
    db.cursor().executemany("""
        INSERT INTO documents (url, module, issue, date, filename, path)
        VALUES (%(url)s, %(module)s, %(issue)s, %(date)s, %(filename)s, %(path)s)
        ON CONFLICT (url) DO UPDATE SET
            issue = excluded.issue, date = excluded.date, filename = excluded.filename,
            path = excluded.path, updated_at = now()""", issues)


def pending(db, limit):
    return db.execute("""
        SELECT * FROM documents
        WHERE status <> 'stored' AND attempts < %s
        ORDER BY date DESC LIMIT %s""", [MAX_ATTEMPTS, limit]).fetchall()


def save_results(db, results):
    db.cursor().executemany("""
        UPDATE documents
        SET status = %(status)s, size = %(size)s, sha256 = %(sha256)s, error = %(error)s,
            attempts = attempts + 1, updated_at = now()
        WHERE url = %(url)s""", results)


def counts(db):
    return db.execute("SELECT status, count(*) FROM documents GROUP BY status").fetchall()
