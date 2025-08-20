-- posts table
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    url TEXT,
    title TEXT,
    source TEXT,
    published_at TIMESTAMP,
    author TEXT,
    summary TEXT,
    image_url TEXT,
    fulltext TEXT,
    text_embedding BLOB,
    image_embedding BLOB,
    labels TEXT
);

-- topics table
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    keywords TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- mapping
CREATE TABLE IF NOT EXISTS post_topics (
    post_id TEXT REFERENCES posts(id),
    topic_id INTEGER REFERENCES topics(id)
);

-- timeseries
CREATE TABLE IF NOT EXISTS topic_timeseries (
    topic_id INTEGER REFERENCES topics(id),
    week DATE,
    count INTEGER,
    momentum FLOAT,
    burst_score FLOAT
);
