# fix_schema_posts.py
import sqlite3, sys

DB = "trends.db"
con = sqlite3.connect(DB)
cur = con.cursor()

# inspect existing columns
cols = [r[1] for r in cur.execute("PRAGMA table_info(posts)")]

# add summary if missing
if "summary" not in cols:
    print("Adding posts.summary …")
    cur.execute("ALTER TABLE posts ADD COLUMN summary TEXT")

# backfill from content if that column exists and summary is NULL/empty
if "content" in cols:
    print("Backfilling summary from content …")
    cur.execute("UPDATE posts SET summary = COALESCE(NULLIF(summary,''), content) WHERE summary IS NULL OR summary=''")

con.commit()
con.close()
print("Done.")
