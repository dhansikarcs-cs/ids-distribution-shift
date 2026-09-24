# Dataset: CICIDS2017 (Friday subset)

## Download

The CICIDS2017 dataset is publicly distributed by the Canadian Institute for
Cybersecurity (CIC), University of New Brunswick:

- Dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
- Direct: the "CSV files in .7z" (or per-day) download

This project uses the **Friday** portion of the dataset (the file commonly named
`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` together with the other
Friday flow-record CSVs), which contains the BENIGN, Bot, DDoS, and PortScan
classes used in the manuscript.

## Why it is not in this repository

- The raw dataset is several gigabytes and exceeds repository size limits.
- It is a public, third-party dataset under its own terms of use.

Place the merged Friday flow records as `cicids2017_friday.csv` in the project
root (next to `code/`) before running the scripts.

## Preprocessing performed (mirrors the manuscript, Section 5.1)

1. Replace infinite values with NaN.
2. Drop rows containing NaN/Inf: 527 rows (0.07% of the raw 703,245) removed;
   no imputation applied.
3. Drop non-numeric columns; keep the 78 numeric features.
4. Binary label: BENIGN (0) vs. ATTACK (1). Final size: 318,237 rows
   (BENIGN 29,452 / Bot 1,956 / DDoS 128,025 / PortScan 158,804).