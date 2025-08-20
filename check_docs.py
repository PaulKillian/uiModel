import sqlite3, os

DB = os.path.abspath("trends.db")
con = sqlite3.connect(DB)

n = con.execute("SELECT COUNT(*) FROM posts WHERE text_embedding IS NOT NULL").fetchone()[0]
print("docs with embeddings:", n)

con.close()
