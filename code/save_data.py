import pandas as pd
import os

cache_dir = r'C:\Users\dhans\.cache\huggingface\hub\datasets--bvsam--cic-ids-2017'
parquet_files = []
for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if f.endswith('.parquet') and 'incomplete' not in f:
            parquet_files.append(os.path.join(root, f))

dfs = []
for pf in parquet_files:
    dfs.append(pd.read_parquet(pf))

combined = pd.concat(dfs, ignore_index=True)
out = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'
combined.to_csv(out, index=False)
print(f'Saved: {len(combined):,} rows, {len(combined.columns)} cols')
print(f'Size: {os.path.getsize(out) / 1024 / 1024:.1f} MB')
