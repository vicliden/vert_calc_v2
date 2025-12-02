import io
import os
import sys

# When Streamlit runs the script from the `app/` directory the repository root
# (where `jump_analysis.py` lives) may not be on sys.path. Make sure the repo
# root is available so imports like `import jump_analysis` work on Render
# and other hosts that change the current working directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import zipfile
import json
from typing import List

import streamlit as st
import pandas as pd

from jump_analysis import analyze_jumps_from_folder, load_accelerometer_folder, to_mag_df

st.set_page_config(page_title="Jump Analyzer", layout="centered")

st.title("Jump Analyzer — phyphox CSVs")

st.markdown(
    "Upload one or more phyphox/Ead/CSV files (or a zip of them). The app will group files by folder-style uploads, call analyze_jumps_from_folder, and show per-jump plots & metrics."
)

uploaded = st.file_uploader("Upload CSV files or a ZIP", accept_multiple_files=True, type=["csv", "zip"])

# helper to write uploaded files to a temporary directory structure
TMP_IN = os.path.join("tmp_upload")
if not os.path.exists(TMP_IN):
    os.makedirs(TMP_IN)

@st.cache_data
def save_uploaded_files(files) -> List[str]:
    """Save uploaded files to a temp folder and return list of folder paths we can analyze.

    If a single zip is uploaded we extract it into a subfolder. If multiple CSVs are uploaded
    we place them into one folder under tmp_upload/.
    """
    os.makedirs(TMP_IN, exist_ok=True)

    if len(files) == 0:
        return []

    # if a single zip, extract to a named folder
    if len(files) == 1 and files[0].name.lower().endswith(".zip"):
        z = files[0]
        target_dir = os.path.join(TMP_IN, os.path.splitext(z.name)[0])
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(z.read())) as zf:
            zf.extractall(target_dir)
        return [target_dir]

    # otherwise save all CSVs into a single folder
    csv_dir = os.path.join(TMP_IN, "csv_upload")
    os.makedirs(csv_dir, exist_ok=True)
    for f in files:
        raw = f.read()
        path = os.path.join(csv_dir, f.name)
        with open(path, "wb") as fh:
            fh.write(raw)
    return [csv_dir]


if uploaded:
    folders = save_uploaded_files(uploaded)
else:
    folders = []

if len(folders) == 0:
    st.info("Upload CSV(s) or a ZIP file with phyphox CSVs to get started.")
    st.stop()

st.sidebar.header("Options")
save_plots = st.sidebar.checkbox("Save plots to results folder", value=False)
window_radius = st.sidebar.number_input("Per-jump window radius (s)", min_value=0.5, max_value=10.0, value=2.0)
rel_peak_thresh = st.sidebar.slider("Relative landing peak threshold", min_value=0.1, max_value=1.0, value=0.5)

# process each folder and show results
for folder in folders:
    st.header(f"Folder: {folder}")

    # If more than one folder exists inside, allow choosing subfolder
    children = [os.path.join(folder, p) for p in os.listdir(folder) if os.path.isdir(os.path.join(folder, p))]
    all_roots = [folder] + children
    chosen = st.selectbox("Choose folder to analyze", options=all_roots, index=0)

    st.write("Loading data and analyzing — this may take a few seconds.")

    try:
        results = analyze_jumps_from_folder(chosen, save_plots=save_plots, result_dir=os.path.join(chosen, "results"), )
    except Exception as e:
        st.error(f"Error while analyzing folder: {e}")
        continue

    if not results:
        st.warning("No jumps detected in this folder.")
        continue

    st.success(f"Found {len(results)} jumps")

    for i, r in enumerate(results, start=1):
        st.subheader(f"Jump {i}")
        cols = st.columns([1, 2])

        # show plot (either returned bytes, or file path if saved)
        plot = r.get("plot")
        if isinstance(plot, (bytes, bytearray)):
            cols[0].image(plot, caption=f"Jump {i}")
        elif isinstance(plot, str) and os.path.exists(plot):
            cols[0].image(plot, caption=f"Jump {i}")

        # metrics
        metrics = {k: v for k, v in r.items() if k not in ("plot", "window_df")}
        # pretty display
        with cols[1]:
            st.json(metrics)
            st.download_button(f"Download metrics JSON (Jump {i})", data=json.dumps(metrics, default=str), file_name=f"jump_{i}_metrics.json", mime="application/json")

        # show the captured window dataframe as a small table and allow CSV download
        df = r.get("window_df")
        if df is not None:
            try:
                display_df = df.copy()
                # convert to string-friendly formats
                display_df["Time (s)"] = display_df["Time (s)"].astype(float)
                csv_buf = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download window CSV (Jump {i})", data=csv_buf, file_name=f"jump_{i}_window.csv", mime="text/csv")
                st.dataframe(display_df)
            except Exception:
                st.write("Could not render window data")

st.markdown("---")
st.caption("Exported plots will be saved into a 'results' subfolder under the chosen folder when 'Save plots' is toggled.")
