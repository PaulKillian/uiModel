Mass UI/UX Ingestion Kit (v2)

Commands to fetch ~1000 posts (PowerShell):
  python ingest_rss.py --feeds feeds_mass.yaml --max 80 --wp-pages 6 --requests --opengraph --no-cutoff

Then run your pipeline:
  python cluster_topics.py --min_topic_size 5
  python trends_momentum.py --window 6
  python categorize_topics.py
  python tag_ui_components.py

Tip: try a single feed first if you want to verify inserts:
  python ingest_rss.py --one https://www.smashingmagazine.com/feed/ --wp-pages 8 --requests --no-cutoff
