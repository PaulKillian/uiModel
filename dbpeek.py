# dbpeek.py
import sqlite3, pandas as pd
con = sqlite3.connect("trends.db")
print("posts:", pd.read_sql_query("select count(*) as n from posts", con))
print(pd.read_sql_query("select title, source, published_at from posts order by published_at desc limit 10", con))
con.close()
