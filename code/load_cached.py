import pandas as pd
import os

cache_dir = r'C:\Users\dhans\.cache\huggingface\hub\datasets--bvsam--cic-ids-2017'

parquet_files = []
for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if f.endswith('.parquet') and 'incomplete' not in f:
            parquet_files.append(os.path.join(root, f))

print('Found parquet files:')
for pf in parquet_files:
    size_mb = os.path.getsize(pf) / 1024 / 1024
    print(f'  {os.path.basename(pf)}: {size_mb:.1f} MB')

dfs = []
for pf in parquet_files:
    df = pd.read_parquet(pf)
    print(f'\n{os.path.basename(pf)}:')
    print(f'  Shape: {df.shape}')
    print(f'  Columns (first 5): {list(df.columns[:5])}')
    label_col = [c for c in df.columns if 'label' in c.lower()]
    if label_col:
        lc = label_col[0]
        print(f'  Labels: {df[lc].value_counts().to_dict()}')
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f'\nCombined shape: {combined.shape}')
label_col = [c for c in combined.columns if 'label' in c.lower()]
if label_col:
    lc = label_col[0]
    print(f'Combined labels:')
    for label, count in combined[lc].value_counts().items():
        print(f'  {label}: {count:,}')
print(f'Total rows: {len(combined):,}')
print(f'Total features: {len(combined.columns)}')
