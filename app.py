"""
Ghodke Insights AI — AI Data Assistant (Advanced)
Single-file Streamlit app.

Features:
- Upload CSV / Excel or load sample dataset
- Cleaning pipeline (standard rules)
- EDA: distributions, correlations, missingness heatmap, top categories
- Charts: histograms, boxplots, correlation heatmap, categorical bars
- Downloadable ZIP containing cleaned.csv, summary.md, and chart PNGs
- Optional LLM adapter for narrative summaries (stub / templated)
- Developer mode for raw logs
- Caching for heavy operations
- Safety confirmation for very large datasets (>200k rows)

Header logo path (local): /mnt/data/79c16058-a7eb-4a74-9c2e-8a599da24b85.png
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import tempfile
import zipfile
import base64
from datetime import datetime
from typing import Tuple, Dict, List, Any

# Page config (required)
st.set_page_config(
    page_title="Ghodke Insights AI — AI Data Assistant",
    page_icon="logo.png",
    layout="wide"
)

# ---------------------------
# Helper & Cached Functions
# ---------------------------

@st.cache_data(ttl=3600)
def load_sample_dataset() -> pd.DataFrame:
    """Return a small sample dataset if user hasn't uploaded a file.
    We'll create a synthetic dataset that illustrates numeric/categorical/date columns.
    """
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "id": np.arange(1, n+1),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="H"),
        "age": rng.integers(18, 70, size=n),
        "salary": (rng.normal(50000, 15000, size=n)).round(2),
        "country": rng.choice(["India", "USA", "UK", "Germany", "France"], size=n),
        "product_category": rng.choice(["A","B","C","D"], size=n),
        "score": rng.normal(0, 1, size=n)
    })
    # Inject some missingness and duplicates
    df.loc[rng.choice(n, size=30, replace=False), "salary"] = np.nan
    df.loc[rng.choice(n, size=20, replace=False), "country"] = None
    df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)  # duplicates
    return df

@st.cache_data
def read_uploaded_file(uploaded) -> pd.DataFrame:
    """Read CSV / Excel into pandas DataFrame."""
    try:
        if uploaded.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded)
        else:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
    except Exception as e:
        # Try with more lenient options
        uploaded.seek(0)
        df = pd.read_csv(uploaded, engine="python", sep=None, error_bad_lines=False)
    return df

def _clean_column_name(name: str) -> str:
    """Normalize column names: strip, lower, replace spaces with underscores."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace("\t", "_")
    )

@st.cache_data
def basic_cleaning_pipeline(df: pd.DataFrame,
                            drop_threshold: float = 0.9,
                            fill_numeric_with: str = "median",
                            drop_constant: bool = True,
                            convert_dates: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleaning rules (general / 'Basic prompt' style):
    - Normalize column names
    - Drop columns with > drop_threshold fraction missing
    - Strip whitespace in string columns
    - Convert obvious date columns
    - Convert object columns with few uniques to category
    - Fill numeric missing values with median/mean/zero as chosen
    - Remove fully-duplicate rows
    - Drop constant columns (optional)
    - Return cleaned df and a small audit dict
    """
    audit = {"original_shape": df.shape, "steps": []}
    df = df.copy()
    # Normalize column names
    orig_cols = df.columns.tolist()
    new_cols = [_clean_column_name(c) for c in orig_cols]
    df.columns = new_cols
    audit["renamed_columns"] = dict(zip(orig_cols, new_cols))
    audit["steps"].append("normalized_column_names")

    # Drop columns with too much missing
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > drop_threshold].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        audit["steps"].append(f"dropped_columns_{to_drop}")

    # Strip whitespace in object columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": None})
    audit["steps"].append("stripped_whitespace_in_object_columns")

    # Convert possible dates (heuristic)
    if convert_dates:
        for c in df.columns:
            if df[c].dtype == "object":
                sample = df[c].dropna().astype(str).head(30).tolist()
                # quick heuristic: presence of '-' or '/' or 'T' or 'AM'/'PM'
                if any(("-" in s or "/" in s or "T" in s or ":" in s) for s in sample):
                    try:
                        df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                        if df[c].notna().sum() > 0:
                            audit["steps"].append(f"converted_to_datetime:{c}")
                    except Exception:
                        pass

    # Convert low-cardinality object columns to category
    for c in df.select_dtypes(include="object").columns:
        if df[c].nunique(dropna=True) / max(1, len(df)) < 0.05:
            df[c] = df[c].astype("category")
            audit["steps"].append(f"converted_to_category:{c}")

    # Drop constant columns
    if drop_constant:
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            df = df.drop(columns=const_cols)
            audit["steps"].append(f"dropped_constant_columns:{const_cols}")

    # Remove duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        audit["steps"].append(f"dropped_{dup_count}_duplicate_rows")

    # Fill numeric missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            if fill_numeric_with == "median":
                val = df[c].median()
            elif fill_numeric_with == "mean":
                val = df[c].mean()
            elif fill_numeric_with == "zero":
                val = 0
            else:
                val = df[c].median()
            df[c] = df[c].fillna(val)
            audit["steps"].append(f"filled_numeric_{c}_with_{fill_numeric_with}")

    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    for c in cat_cols:
        if df[c].isna().any():
            try:
                mode_val = df[c].mode(dropna=True).iloc[0]
            except Exception:
                mode_val = ""
            df[c] = df[c].fillna(mode_val)
            audit["steps"].append(f"filled_categorical_{c}_with_mode")

    audit["cleaned_shape"] = df.shape
    return df, audit

@st.cache_data
def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics used for narrative / summary.md"""
    stats = {}
    stats["n_rows"], stats["n_cols"] = df.shape
    stats["missingness"] = df.isna().sum().to_dict()
    stats["dtypes"] = df.dtypes.apply(lambda x: str(x)).to_dict()
    # numeric summary
    num = df.select_dtypes(include=[np.number])
    stats["numeric_describe"] = num.describe().to_dict()
    # top categories for object/categorical columns
    cat_top = {}
    for c in df.select_dtypes(include=["object", "category"]).columns:
        top = df[c].value_counts(dropna=True).head(5).to_dict()
        cat_top[c] = top
    stats["top_categories"] = cat_top
    # correlation (pearson) for numeric
    if num.shape[1] >= 2:
        stats["correlation"] = num.corr().to_dict()
    else:
        stats["correlation"] = {}
    return stats

# ---------------------------
# Plotting Functions
# ---------------------------
def safe_sample_for_plot(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """Return a random sample up to max_rows for plotting to keep UI responsive."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)

def plot_histogram(df: pd.DataFrame, column: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[column].dropna(), kde=False, ax=ax)
    ax.set_title(f"Histogram: {column}")
    plt.tight_layout()
    return fig

def plot_boxplot(df: pd.DataFrame, column: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[column].dropna(), ax=ax)
    ax.set_title(f"Boxplot: {column}")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap (numeric)")
    plt.tight_layout()
    return fig

def plot_missingness_heatmap(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    ax.set_title("Missingness Heatmap")
    plt.tight_layout()
    return fig

def plot_categorical_bar(df: pd.DataFrame, column: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df[column].value_counts(dropna=True).head(20)
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_title(f"Top categories: {column}")
    plt.tight_layout()
    return fig

# ---------------------------
# LLM Adapter (deterministic / stub)
# ---------------------------

def generate_narrative_summary(summary_stats: Dict[str, Any],
                               sample_rows: pd.DataFrame,
                               top_plots: List[str],
                               model: str = "local") -> str:
    """
    Adapter for narrative summary.
    - If model == 'openai' (or 'gpt-...') or OPENAI_API_KEY env var present, this is where you would call the LLM API.
      For safety and portability this function currently uses a deterministic template if model != 'openai' or API key absent.
    - Keep this function isolated so user can replace the stub with real API integration.

    NOTE: To integrate OpenAI, detect OPENAI_API_KEY via os.environ.get('OPENAI_API_KEY') and call their API here.
    """
    # Check for environment key (user can set OPENAI_API_KEY). We don't call internet from this template.
    openai_key = os.environ.get("OPENAI_API_KEY", None)
    if model == "openai" or openai_key:
        # PLACEHOLDER: Insert OpenAI / other LLM API call here.
        # Example (pseudo):
        # prompt = build_prompt_from_stats(summary_stats, sample_rows, top_plots)
        # client = OpenAI(api_key=openai_key)
        # response = client.create(prompt=prompt, ...)
        # return response.text
        return ("[LLM mode requested] -- OpenAI / external LLM integration point.\n\n"
                "This area is intentionally a stub. To enable, set environment variable OPENAI_API_KEY\n"
                "and replace this block with an actual API call (openai.ChatCompletion.create or similar).")
    # Deterministic template-based summary (3-6 bullets)
    bullets = []
    bullets.append(f"- Dataset contains **{summary_stats.get('n_rows', '?')} rows** and **{summary_stats.get('n_cols', '?')} columns**.")
    # Missingness quick note
    missing = summary_stats.get("missingness", {})
    if missing:
        total_missing_cols = sum(1 for v in missing.values() if v and v > 0)
        bullets.append(f"- Missing values present in **{total_missing_cols} columns**. Most common numeric missingness shown in summary.")
    # Top numeric insights
    numeric_desc = summary_stats.get("numeric_describe", {})
    if numeric_desc:
        some_numeric = list(numeric_desc.keys())[:3]  # top 3 numeric cols available
        for col in some_numeric:
            col_stats = numeric_desc.get(col, {})
            mean = col_stats.get("mean", None)
            std = col_stats.get("std", None)
            if mean is not None:
                bullets.append(f"- Numeric column **{col}** has mean ≈ {float(mean):.2f} and std ≈ {float(std):.2f}.")
    # Top categorical
    top_cats = summary_stats.get("top_categories", {})
    if top_cats:
        sample_cols = list(top_cats.keys())[:2]
        for c in sample_cols:
            top = list(top_cats[c].items())[:3]
            top_str = ", ".join([f"{k} ({v})" for k, v in top])
            bullets.append(f"- Column **{c}** top values: {top_str}.")
    # Correlation hint
    corr = summary_stats.get("correlation", {})
    if corr:
        bullets.append("- Numeric correlation matrix computed; review correlation heatmap for potential multicollinearity.")
    bullets.append(f"- Generated {len(top_plots)} plots (histograms/boxplots/categorical) to inspect distribution and outliers.")
    return "\n".join(bullets)

# ---------------------------
# ZIP creation for download
# ---------------------------

def create_results_zip(clean_df: pd.DataFrame,
                       summary_text: str,
                       figures: List[Tuple[str, plt.Figure]]) -> bytes:
    """
    Create a ZIP bytes with:
    - cleaned.csv
    - summary.md
    - chart PNGs (from figures list of (filename, fig))
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # cleaned.csv
        csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
        zf.writestr("cleaned.csv", csv_bytes)
        # summary
        zf.writestr("summary.md", summary_text.encode("utf-8"))
        # figures
        for fname, fig in figures:
            img_bytes = io.BytesIO()
            fig.savefig(img_bytes, format="png", bbox_inches="tight")
            img_bytes.seek(0)
            zf.writestr(f"charts/{fname}.png", img_bytes.read())
            plt.close(fig)
    bio.seek(0)
    return bio.read()

# ---------------------------
# UI: Header, Sidebar (Options), Upload, Preview, Run, Results
# ---------------------------

# Header
st.title("Ghodke Insights AI — AI Data Assistant (Advanced)")
col1, col2 = st.columns([1, 4])
with col1:
    # show logo if exists
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
with col2:
    st.markdown(
        """
        **Upload any CSV or Excel file** and get a fast cleaning pipeline, EDA, visualizations,
        and a ZIP download containing `cleaned.csv`, `summary.md`, and chart images.
        """
    )

# Sidebar - Options
st.sidebar.header("Options")
use_sample = st.sidebar.button("Load sample dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xls", "xlsx"])
developer_mode = st.sidebar.checkbox("Developer mode (show logs & audits)", value=False)
use_llm = st.sidebar.checkbox("Use LLM to generate narrative summary", value=False)
llm_model_choice = st.sidebar.selectbox("LLM model adapter", options=["local (template)", "openai"], index=0)
max_numeric_plots = st.sidebar.slider("Max numeric columns to plot", 1, 12, 6)
fill_numeric_with = st.sidebar.selectbox("Fill numeric missing with", options=["median", "mean", "zero"], index=0)
drop_threshold = st.sidebar.slider("Drop columns with >% missing", 50, 100, 90) / 100.0
sample_for_plots = st.sidebar.slider("Max rows to sample for plotting", 1000, 10000, 5000)

# Upload Section
st.header("Upload")
uploaded_df = None
if use_sample:
    df = load_sample_dataset()
    st.success("Sample dataset loaded.")
elif uploaded_file is not None:
    try:
        df = read_uploaded_file(uploaded_file)
        st.success(f"Loaded `{uploaded_file.name}` ({df.shape[0]} rows × {df.shape[1]} cols).")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        df = None
else:
    st.info("No file uploaded. Click 'Load sample dataset' from the sidebar or upload a CSV/Excel file.")
    df = None

# Preview Section
st.header("Preview")
if df is not None:
    with st.expander("Show first 10 rows"):
        st.dataframe(df.head(10))
    with st.expander("Show dataframe info"):
        buf = io.StringIO()
        try:
            df.info(buf=buf)
            s = buf.getvalue()
            st.text(s)
        except Exception:
            st.write(df.dtypes)
else:
    st.write("Awaiting dataset...")

# Run Section: Clean + EDA
st.header("Run")
if df is not None:
    # Show small dataset stats
    st.write(f"**Current shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # Confirm if dataset very large
    large_dataset = df.shape[0] > 200_000
    proceed = st.button("Run cleaning & EDA")
    if large_dataset and proceed:
        st.warning("Dataset is very large (>200,000 rows). Creating a ZIP may be slow and resource-heavy.")
        confirm_large = st.checkbox("I understand and want to continue (confirm for large dataset).")
        if not confirm_large:
            st.stop()

    if proceed:
        with st.spinner("Running cleaning pipeline and computing EDA..."):
            clean_df, audit = basic_cleaning_pipeline(
                df,
                drop_threshold=drop_threshold,
                fill_numeric_with=fill_numeric_with,
                drop_constant=True,
                convert_dates=True
            )
            summary_stats = compute_summary_stats(clean_df)
            sampled = safe_sample_for_plot(clean_df, max_rows=sample_for_plots)

        st.success("Cleaning & EDA complete.")
        if developer_mode:
            st.subheader("Developer logs / audit")
            st.json(audit)

        # Results Section
        st.header("Results")

        # Summary narrative
        if use_llm:
            model_key = "openai" if llm_model_choice == "openai" else "local"
            narrative = generate_narrative_summary(summary_stats, sampled.head(5), top_plots=[], model=model_key)
            st.subheader("Narrative Summary")
            st.markdown(narrative)
        else:
            st.subheader("Deterministic Summary (template)")
            template = generate_narrative_summary(summary_stats, sampled.head(5), top_plots=[], model="local")
            st.markdown(template)

        # Show summary statistics
        st.subheader("Summary Statistics")
        st.write(f"- Rows: {summary_stats.get('n_rows')}, Columns: {summary_stats.get('n_cols')}")
        st.write("Top categories (sample):")
        st.write(summary_stats.get("top_categories"))

        # Charts
        st.subheader("Charts")
        charts_to_save = []  # list of (filename, fig)
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = clean_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Limit numeric columns plotted
        numeric_cols_plot = numeric_cols[:max_numeric_plots]

        # Missingness heatmap
        fig_miss = plot_missingness_heatmap(clean_df if len(clean_df) <= 2000 else sampled)
        st.pyplot(fig_miss)
        charts_to_save.append(("missingness_heatmap", fig_miss))

        # Correlation heatmap (if enough numeric)
        if len(numeric_cols) >= 2:
            fig_corr = plot_correlation_heatmap(sampled[numeric_cols_plot])
            st.pyplot(fig_corr)
            charts_to_save.append(("correlation_heatmap", fig_corr))

        # Histograms and boxplots for numeric columns
        st.markdown("**Numeric column distributions (sampled up to {})**".format(sample_for_plots))
        col1, col2 = st.columns(2)
        plotted = 0
        for c in numeric_cols_plot:
            if plotted % 2 == 0:
                with col1:
                    fig_h = plot_histogram(sampled, c)
                    st.pyplot(fig_h)
                    charts_to_save.append((f"hist_{c}", fig_h))
            else:
                with col2:
                    fig_b = plot_boxplot(sampled, c)
                    st.pyplot(fig_b)
                    charts_to_save.append((f"box_{c}", fig_b))
            plotted += 1

        # Categorical bar charts (top categories)
        if cat_cols:
            st.markdown("**Categorical top categories**")
            for c in cat_cols[:6]:
                fig_cat = plot_categorical_bar(sampled, c)
                st.pyplot(fig_cat)
                charts_to_save.append((f"cat_{c}", fig_cat))

        # Show cleaned dataframe preview & download cleaned csv
        st.subheader("Cleaned Data Preview & Download")
        with st.expander("Show cleaned data (first 50 rows)"):
            st.dataframe(clean_df.head(50))
        csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned.csv", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")

        # Create summary.md content
        summary_md_lines = [
            f"# Dataset summary - generated {datetime.utcnow().isoformat()} UTC",
            "",
            f"- Original shape: {audit.get('original_shape')}",
            f"- Cleaned shape: {audit.get('cleaned_shape')}",
            "",
            "## Cleaning steps (audit):",
            "",
        ]
        for s in audit.get("steps", []):
            summary_md_lines.append(f"- {s}")
        summary_md_lines.append("")
        summary_md_lines.append("## Narrative summary")
        summary_md_lines.append("")
        # Use narrative variable if exists else template
        if use_llm:
            summary_md_lines.append(narrative)
        else:
            summary_md_lines.append(template)
        summary_text = "\n".join(summary_md_lines)

        # ZIP export (cleaned.csv, summary.md, charts)
        st.subheader("Export results (ZIP)")
        make_zip = st.button("Create results ZIP")
        if make_zip:
            if clean_df.shape[0] > 200_000:
                st.warning("You confirmed earlier. Creating ZIP for large dataset may take time and memory.")
            with st.spinner("Rendering charts and creating ZIP..."):
                # Ensure charts_to_save figs are re-created as saveables (some backends close after st.pyplot)
                # Recreate a minimal set of plots for ZIP to be safe
                figures_for_zip = []
                # recreate missingness as png
                fig_miss2 = plot_missingness_heatmap(clean_df if len(clean_df) <= 2000 else sampled)
                figures_for_zip.append(("missingness_heatmap", fig_miss2))
                if len(numeric_cols) >= 2:
                    fig_corr2 = plot_correlation_heatmap(sampled[numeric_cols_plot])
                    figures_for_zip.append(("correlation_heatmap", fig_corr2))
                for c in numeric_cols_plot:
                    figures_for_zip.append((f"hist_{c}", plot_histogram(sampled, c)))
                    figures_for_zip.append((f"box_{c}", plot_boxplot(sampled, c)))
                for c in cat_cols[:6]:
                    figures_for_zip.append((f"cat_{c}", plot_categorical_bar(sampled, c)))

                zip_bytes = create_results_zip(clean_df, summary_text, figures_for_zip)
                st.success("ZIP created.")
                st.download_button("Download results.zip", data=zip_bytes, file_name="ghodke_insights_results.zip", mime="application/zip")

else:
    st.info("Upload or load a dataset to enable cleaning & EDA.")

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.markdown(
    "If you want to customize or plug in your own LLM, edit `generate_narrative_summary()` "
    "and replace the stub with your API calls. Keep API keys out of source control!"
)

# ---------------------------
# Packaging & Deployment Notes (commented block)
# ---------------------------
# Below are developer notes & examples for setting OPENAI_API_KEY and deployment.
#
# 1) Setting OPENAI_API_KEY locally (Linux / macOS):
#    export OPENAI_API_KEY="sk-...."
#
#    On Windows (PowerShell):
#    $env:OPENAI_API_KEY="sk-...."
#
# 2) Example (pseudo) to integrate OpenAI in generate_narrative_summary():
#    import openai
#    openai.api_key = os.environ.get("OPENAI_API_KEY")
#    prompt = "Summarize dataset: ..."  # build from summary_stats & sample_rows
#    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user", "content": prompt}])
#    return resp.choices[0].message.content
#
# 3) Deploy to Streamlit Cloud:
#    - Create repository with this file app.py, requirements.txt, and the logo under /mnt/data path
#    - On Streamlit Cloud, set the environment variable OPENAI_API_KEY through app settings (if required)
#    - Add secrets via Streamlit secrets manager for API keys
#
# 4) Dockerfile example:
#    FROM python:3.11-slim
#    WORKDIR /app
#    COPY . /app
#    RUN pip install --no-cache-dir -r requirements.txt
#    EXPOSE 8501
#    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#
# 5) Example requirements.txt minimal:
#    streamlit
#    pandas
#    numpy
#    matplotlib
#    seaborn
#    openpyxl  # if you want Excel support
#
# 6) Security & best practices:
#    - Never commit API keys to repo. Use environment variables or secrets manager.
#    - For large datasets consider processing offline or via chunking.
#    - For production LLM calls, add rate-limiting and retry logic.
#
# End of file

