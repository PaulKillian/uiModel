import os
import sqlite3

DB = os.path.abspath("trends.db")

PRAGMAS = [
    # better durability/perf defaults for app databases
    "PRAGMA journal_mode=WAL;",       # concurrent readers, fewer fsyncs
    "PRAGMA synchronous=NORMAL;",     # good balance for WAL
    "PRAGMA temp_store=MEMORY;",
]

INDEXES = [
    # posts: speed time and source queries
    "CREATE INDEX IF NOT EXISTS idx_posts_published_at ON posts(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_posts_source       ON posts(source);",

    # post_topics: avoid duplicate rows + speed lookups
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_post_topics        ON post_topics(post_id, topic_id);",
    "CREATE INDEX IF NOT EXISTS idx_post_topics_post         ON post_topics(post_id);",
    "CREATE INDEX IF NOT EXISTS idx_post_topics_topic        ON post_topics(topic_id);",

    # topic_timeseries: trend rollups
    "CREATE INDEX IF NOT EXISTS idx_timeseries_topic_week    ON topic_timeseries(topic_id, week);",
    "CREATE INDEX IF NOT EXISTS idx_timeseries_week          ON topic_timeseries(week);",
]

def exec_many(conn, statements):
    cur = conn.cursor()
    for sql in statements:
        cur.execute(sql)
    conn.commit()

def main():
    if not os.path.exists(DB):
        raise SystemExit(f"Database not found at {DB}. Run db_setup.py first.")

    conn = sqlite3.connect(DB)
    try:
        # sanity: ensure required tables exist
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"posts", "post_topics", "topic_timeseries"}
        missing = required - tables
        if missing:
            raise SystemExit(f"Missing tables: {missing}. Create schema first (run db_setup.py).")

        # pragmas
        exec_many(conn, PRAGMAS)
        # indexes
        exec_many(conn, INDEXES)

        # optional: gather stats so planner uses indexes well
        conn.execute("ANALYZE;")
        conn.commit()

        print("✅ Pragmas applied and indexes ensured.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
