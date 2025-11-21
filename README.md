# Ghodke Insights AI — AI Data Assistant

**AI Data Assistant — Streamlit app for automated data cleaning, exploratory data analysis (EDA), visualizations, and downloadable results.**

---

## What this project does

- Upload CSV / Excel or load a sample dataset.
- Run a cleaning pipeline (normalize column names, drop high-missing columns, fill missing numeric values, remove duplicates, convert types).
- Compute summary statistics, missingness, correlations and top categories.
- Produce charts (histograms, boxplots, correlation heatmap, categorical bar charts).
- Optionally generate a deterministic narrative summary (local template) or integrate an LLM (via `OPENAI_API_KEY`).
- Export a ZIP containing `cleaned.csv`, `summary.md`, and chart PNGs.

---

## Files in this repo

- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `logo.png` — app favicon and header logo (place your logo here)
- `.gitignore` — recommended (see below)
- `README.md` — this file

> If you have the provided logo in your environment, you can copy it from:
> `/mnt/data/b0c70198-9963-4308-98b5-c31f3c44752a.png`
>
> Save it to this repo as `logo.png` before deploy.

---

## Quick local setup

1. Create & activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
