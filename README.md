# Evaluating Machine Learning-Based Intrusion Detection Models Under Distribution Shift

Controlled experiments on the CICIDS2017 dataset showing that conventional random-split evaluation can severely overestimate the operational effectiveness of ML-based intrusion detection models under distribution shift.

**Key results:**
- Random-split F1 of 0.9997 collapses to **0.0000** (Random Forest) under a joint session + attack-family shift.
- A matched-size control (same 13,736 training samples, same distribution) recovers **0.9992**, proving the degradation is distributional, not a data-scarcity effect.
- A single-class covariate-shift control isolates **unseen attack classes** (not feature displacement) as the primary driver of collapse.
- Tree-based models retain ranking signal (ROC-AUC 0.82-0.85) but need far-below-default thresholds; recalibration is a partial, bounded remedy.

## Timeline

The project was carried out from **July 2026 to September 2026**. This repository was assembled and published in **September 2026, after the manuscript was completed**: all experiment scripts, outputs, results, and figures were produced and recorded during the evaluation period (July-September 2026) and are archived here as the reproducible record. The paper source and the arXiv-formatted LaTeX bundle are included under `paper/` and `arxiv/`.

## Repository layout

```
├── paper/            Manuscript: paper.md (source) and paper.pdf (formatted)
├── arxiv/            arXiv-ready LaTeX: main.tex, fig1-8.png, submission_fields.txt
├── code/             All experiment, plotting, verification, and PDF-building scripts
├── results/          All experiment outputs: figures, JSON result files, summary table
├── data/             Dataset notes and download instructions (dataset not bundled)
├── LICENSE           MIT
└── README.md
```

## Reproducibility

1. **Dataset:** download the CICIDS2017 dataset (see `data/README.md`) and place the Friday (PHI) flow records as `cicids2017_friday.csv` next to the scripts.
2. **Install:** `pip install numpy pandas scikit-learn xgboost matplotlib`
3. **Run experiments:** `python code/experiments_v3.py` (Experiment B4, threshold sweep, Wasserstein distances) and `python code/experiments_v4.py` (fine threshold sweep).
4. **Generate figures:** `python code/generate_plots_v3.py` and `python code/generate_plots_final.py`.
5. **Verify every reported number:** `python code/verify_numbers.py` (155 checks against the experiment logs) and `python code/verify_methodology.py`.
6. **Build the paper PDF:** `python code/make_pdf.py`; **build the arXiv LaTeX bundle:** `python code/build_arxiv.py`.

All models use fixed hyperparameters with `random_state=42` for reproducibility.

## Results

`results/all_results.json`, `results/supplementary_results.json`, and `results/threshold_sweep_curve.json` contain the full verified numbers used in the manuscript (Experiments A, B, B2, B4, C, D, Wasserstein distances, ROC-AUC, and the threshold sweep). The 8 figures referenced in the manuscript are in `results/`.

## Dataset

The dataset is **not committed** to this repository because it is large and publicly distributed. See `data/README.md` for download instructions and the exact preprocessing steps.

## Manuscript

- **Title:** Evaluating Machine Learning-Based Intrusion Detection Models Under Distribution Shift
- **Author:** Dhansika.R (Independent Researcher)
- **Format:** preprint (arXiv-style), also formatted for peer-review submission
- **AI assistance disclosure:** Generative AI / AI-assisted tools (including the tool *opencode*) were used as guided assistance for scripting, figure generation, and prose editing; the author made all high-level experimental design decisions and reviewed/validated every reported result. Full disclosure is in the manuscript's Declarations section.

## License

MIT - see [LICENSE](LICENSE).

## Citation

If you use this work, please cite the manuscript:

```
Dhansika.R. Evaluating Machine Learning-Based Intrusion Detection Models Under
Distribution Shift. 2026. (preprint)
```