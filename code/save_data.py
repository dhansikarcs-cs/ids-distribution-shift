import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.environ.get(
    'HF_CICIDS_CACHE',
    os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub',
                 'datasets--bvsam--cic-ids-2017'),
)
parquet_files = []
for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if f.endswith('.parquet') and 'incomplete' not in f:
            parquet_files.append(os.path.join(root, f))

dfs = []
for pf in parquet_files:
    dfs.append(pd.read_parquet(pf))

combined = pd.concat(dfs, ignore_index=True)
out = os.environ.get(
    'CICIDS_FRIDAY_CSV', os.path.join(BASE, 'data', 'cicids2017_friday.csv')
)
os.makedirs(os.path.dirname(out), exist_ok=True)
combined.to_csv(out, index=False)
print(f'Saved: {len(combined):,} rows, {len(combined.columns)} cols')
print(f'Size: {os.path.getsize(out) / 1024 / 1024:.1f} MB')
