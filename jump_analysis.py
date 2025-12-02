import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import json
import io
import sys
import re

if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile

GRAVITY = 9.81  # m/s^2


# ----------------------------
# 1. DATA LOADING / PREP
# ----------------------------
def load_accelerometer_folder(folder_name: str) -> pd.DataFrame:
    """
    Reads accelerometer CSV(s) from a folder, single file, or ZIP archive.

    Expected columns in CSV:
        "Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"

    Returns
    -------
    accel_df : pd.DataFrame with columns:
        Time (s)
        X (m/s^2), Y (m/s^2), Z (m/s^2)
        Magnitude (m/s^2)
    """
    required_columns = ["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"]

    def _clean_text(s: str) -> str:
        # normalize unicode minus to ASCII hyphen
        s = s.replace("\u2212", "-")
        # convert '×10^n', '× 10^n', 'x10^n', 'x 10 n' etc -> 'eN'
        s = re.sub(r'[×xX]\s*10\^?\s*([+\-−]?\d+)', r'e\1', s)
        # also accept forms like '5.7×10-4' (no caret)
        s = re.sub(r'([0-9])\s*[×xX]\s*10\s*([+\-−]?\d+)', r'\1e\2', s)
        return s

    def _try_parse_bytes(b: bytes) -> pd.DataFrame | None:
        # try utf-8 first, then latin-1
        for enc in ("utf-8", "latin-1"):
            try:
                txt = b.decode(enc)
            except Exception:
                continue
            txt = _clean_text(txt)
            buf = io.StringIO(txt)
            try:
                # let pandas sniff delimiter
                df = pd.read_csv(buf, sep=None, engine="python")
            except Exception:
                # last resort: try comma
                buf.seek(0)
                try:
                    df = pd.read_csv(buf, sep=",")
                except Exception:
                    continue
            # flexible column matching
            cols = list(df.columns)
            def find_col(preds):
                for c in cols:
                    low = re.sub(r'[^a-z0-9]+', '', c.lower())
                    if all(p in low for p in preds):
                        return c
                return None
            found = {}
            found_time = find_col(["time"])
            found_x = find_col(["x", "m", "s", "2"]) or find_col(["x"])
            found_y = find_col(["y", "m", "s", "2"]) or find_col(["y"])
            found_z = find_col(["z", "m", "s", "2"]) or find_col(["z"])
            if found_time and found_x and found_y and found_z:
                # rename columns to canonical names
                rename_map = {
                    found_time: "Time (s)",
                    found_x: "X (m/s^2)",
                    found_y: "Y (m/s^2)",
                    found_z: "Z (m/s^2)",
                }
                df = df.rename(columns=rename_map)
                # Ensure numeric types for x,y,z and time
                try:
                    df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce")
                    df["X (m/s^2)"] = pd.to_numeric(df["X (m/s^2)"], errors="coerce")
                    df["Y (m/s^2)"] = pd.to_numeric(df["Y (m/s^2)"], errors="coerce")
                    df["Z (m/s^2)"] = pd.to_numeric(df["Z (m/s^2)"], errors="coerce")
                except Exception:
                    return None
                # drop rows where any required numeric column is NaN
                df = df.dropna(subset=["Time (s)", "X (m/s^2)", "Y (m/s^2)", "Z (m/s^2)"])
                return df
        return None

    accel_df = pd.DataFrame()

    # handle zip file
    if os.path.isfile(folder_name) and zipfile.is_zipfile(folder_name):
        with zipfile.ZipFile(folder_name) as zf:
            # search recursively in zip namelist for csvs
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    try:
                        b = zf.read(member)
                    except Exception:
                        continue
                    parsed = _try_parse_bytes(b)
                    if parsed is not None:
                        accel_df = parsed
                        break
        if accel_df.empty:
            raise FileNotFoundError(f"No valid accelerometer CSV found inside ZIP {folder_name!s}")

    # single csv file
    elif os.path.isfile(folder_name) and folder_name.lower().endswith(".csv"):
        with open(folder_name, "rb") as fh:
            b = fh.read()
        parsed = _try_parse_bytes(b)
        if parsed is None:
            raise ValueError(f"CSV file {folder_name!s} does not contain required columns or is unreadable.")
        accel_df = parsed

    # directory (search recursively)
    elif os.path.isdir(folder_name):
        for root, _, files in os.walk(folder_name):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    path = os.path.join(root, fname)
                    with open(path, "rb") as fh:
                        b = fh.read()
                    parsed = _try_parse_bytes(b)
                    if parsed is not None:
                        accel_df = parsed
                        break
            if not accel_df.empty:
                break
        if accel_df.empty:
            raise FileNotFoundError(f"No valid accelerometer CSV found in {folder_name!s}")

    else:
        raise ValueError(f"{folder_name!s} is not a file, zip, or directory.")

    # compute magnitude column as before
    accel_df["Magnitude (m/s^2)"] = np.sqrt(
        accel_df["X (m/s^2)"].astype(float) ** 2
        + accel_df["Y (m/s^2)"].astype(float) ** 2
        + accel_df["Z (m/s^2)"].astype(float) ** 2
    )

    return accel_df

def to_mag_df(accel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Slim dataframe -> just time + magnitude.
    """
    out = pd.DataFrame({
        "Time (s)": accel_df["Time (s)"].to_numpy(),
        "Magnitude": accel_df["Magnitude (m/s^2)"].to_numpy(),
    })
    return out

# ----------------------------
# 2. SIGNAL UTILITIES
# ----------------------------

def estimate_baseline(time_s: np.ndarray,
                      mag: np.ndarray,
                      window_s: float = 0.5) -> float:
    """
    Estimate "rest" baseline for a segment by finding the lowest-variance window.

    We slide a window of ~window_s seconds and pick
    the chunk with the smallest std (most still).
    Return its mean.
    """
    time_s = np.asarray(time_s)
    mag = np.asarray(mag)

    # how many samples in the window?
    dt = np.mean(np.diff(time_s))
    if not np.isfinite(dt) or dt <= 0:
        return float(np.mean(mag))

    window_n = max(3, int(window_s / dt))

    stds = []
    means = []
    for i in range(len(mag) - window_n):
        chunk = mag[i:i + window_n]
        stds.append(np.std(chunk))
        means.append(np.mean(chunk))

    if len(stds) == 0:
        # too short, fallback
        return float(np.mean(mag))

    stds = np.asarray(stds)
    means = np.asarray(means)

    best_idx = int(np.argmin(stds))
    baseline = float(means[best_idx])
    return baseline


def identify_landings(accel_mag: np.ndarray,
                      time_s: np.ndarray,
                      min_sep_seconds: float = 2.0,
                      rel_height_threshold: float = 0.5) -> list[int]:
    """
    Find likely landings across entire recording.

    Heuristic:
    - find all local peaks
    - sort by height desc
    - greedily keep the tallest peaks that are spaced >= min_sep_seconds apart
    - drop weak peaks (< rel_height_threshold * global max)
    - return final peak indices sorted in time

    Returns
    -------
    landing_indices_sorted : list[int]
        indices (into accel_mag/time_s arrays)
    """

    a = np.asarray(accel_mag)
    t = np.asarray(time_s)

    max_peak = np.max(a)

    all_peaks, _ = find_peaks(a)

    peak_heights = a[all_peaks]
    order = np.argsort(peak_heights)[::-1]  # tallest first
    sorted_peaks = all_peaks[order]

    kept_peaks = []
    for p in sorted_peaks:
        t = np.array(t)
        tp = t[p]  # time of this candidate peak
        too_close = False
        for kp in kept_peaks:
            if abs(tp - t[kp]) < min_sep_seconds:
                too_close = True
                break
        if not too_close:
            kept_peaks.append(p)

    kept_peaks = np.array(kept_peaks)
    # use the provided relative threshold parameter when filtering weak peaks
    filtered_peaks = [peak for peak in kept_peaks if a[peak] > max_peak * rel_height_threshold]
    filtered_peaks.sort()

    return filtered_peaks


def split_into_jump_windows(mag_df: pd.DataFrame,
                            window_radius_s: float = 2.0) -> list[pd.DataFrame]:
    """
    Cut the full signal into per-jump windows around each landing.

    For each detected landing index i:
      keep rows where time is within [t_i - window_radius_s, t_i + window_radius_s]

    Returns list of small dataframes with columns:
      Time (s), Magnitude
    """
    t = mag_df["Time (s)"].to_numpy()
    a = mag_df["Magnitude"].to_numpy()

    landing_idxs = identify_landings(a, t)

    jumps_dfs = []
    for i in landing_idxs:
        center = float(t[i])
        mask = (mag_df["Time (s)"] >= center - window_radius_s) & (mag_df["Time (s)"] <= center + window_radius_s)
        jumps_dfs.append(mag_df[mask].copy())
    return jumps_dfs


def zero_gravity(df_jump: pd.DataFrame) -> pd.DataFrame:
    """
    For a single jump window, estimate the standing baseline and subtract it.

    Adds column:
      Magnitude_zeroed  (m/s^2)  ~ acceleration relative to 'still'
    """
    base = estimate_baseline(df_jump["Time (s)"].to_numpy(),
                             df_jump["Magnitude"].to_numpy())
    out = df_jump.copy()
    out["Magnitude_zeroed"] = out["Magnitude"] - base
    return out


# ----------------------------
# 3. TAKEOFF / LANDING DETECTION
# ----------------------------

def detect_takeoff_v(time_s: np.ndarray,
                     mag_zeroed: np.ndarray,
                     lookback_s: float = 1.0,
                     extra_offset_samples: int = 5):
    """
    Simple heuristic:

    1. landing_idx = argmax(accel)  (biggest spike in this window)
    2. search ~lookback_s seconds before landing
    3. find local peaks in that search window
       -> choose the tallest peak in that range as "push-off"
    4. return (takeoff_idx, landing_idx), with a small offset
       because contact loss tends to happen just *after* the max push peak

    Returns
    -------
    takeoff_idx : int
    landing_idx : int
    """
    t = np.asarray(time_s)
    a = np.asarray(mag_zeroed)

    landing_idx = int(np.argmax(a))

    # how many samples ~ lookback_s ? Be robust if dt is invalid
    dt = np.mean(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        start_idx = 0
    else:
        lookback_n = int(round(lookback_s / dt))
        start_idx = max(0, landing_idx - lookback_n)

    # peaks BEFORE landing
    local_seg = a[start_idx:landing_idx]
    peaks, _ = find_peaks(local_seg)

    if len(peaks) == 0:
        return None, landing_idx

    # among these peaks, pick the one with highest accel
    best_local_peak = peaks[np.argmax(local_seg[peaks])]
    takeoff_idx = start_idx + best_local_peak + extra_offset_samples

    # clip if we ran past landing
    takeoff_idx = min(takeoff_idx, landing_idx)

    # ensure indices are ints and within range
    takeoff_idx = int(min(max(start_idx, takeoff_idx), landing_idx))
    return takeoff_idx, int(landing_idx)

def make_jump_graph(df_jump: pd.DataFrame,
                    takeoff_idx: int | None,
                    landing_idx: int,
                    title: str = "Jump Analysis",
                    save_path: str | None = None) -> str | bytes:
    """
    Create a matplotlib plot of the jump window with takeoff/landing marked,
    and save it as a PNG instead of showing it interactively.

    Parameters
    ----------
    df_jump : pd.DataFrame
        Per-jump dataframe with columns "Time (s)" and "Magnitude_zeroed".
    takeoff_idx : int | None
        Index of detected takeoff (or None if not found).
    landing_idx : int
        Index of detected landing.
    title : str
        Plot title.
    save_path : str | None
        If provided, save the PNG to this file path and return the path.
        If None, return PNG bytes (in-memory PNG).

    Returns
    -------
    str | bytes
        If `save_path` provided, returns that path (str). Otherwise returns PNG
        bytes (bytes) containing the image data.
    """
    import io
    import zipfile
    import matplotlib.pyplot as plt

    t = df_jump["Time (s)"].to_numpy()
    a = df_jump["Magnitude_zeroed"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, a, label="Magnitude_zeroed (m/s²)")
    if takeoff_idx is not None:
        # guard index bounds
        if 0 <= takeoff_idx < len(t):
            ax.axvline(t[takeoff_idx], color='g', linestyle='--', label='Takeoff')
    if 0 <= landing_idx < len(t):
        ax.axvline(t[landing_idx], color='r', linestyle='--', label='Landing')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude_zeroed (m/s²)")
    ax.legend()
    ax.grid()

    # Save to file or to in-memory bytes and return the result
    if save_path:
        fig.savefig(save_path, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    data = buf.getvalue()
    buf.close()
    plt.close(fig)
    return data


def compute_air_metrics(time_s: np.ndarray,
                        takeoff_idx: int | None,
                        landing_idx: int) -> dict:
    """
    airtime and jump height using flight-time method:

        airtime = t_land - t_takeoff
        height  = g * airtime^2 / 8

    If takeoff_idx is None, metrics will be None.
    """
    t = np.asarray(time_s)

    if takeoff_idx is None or takeoff_idx >= landing_idx:
        return {
            "takeoff_time_s": None,
            "landing_time_s": float(t[landing_idx]),
            "airtime_s": None,
            "height_m": None,
            "height_cm": None,
        }

    airtime = float(t[landing_idx] - t[takeoff_idx])
    height_m = GRAVITY * (airtime ** 2) / 8.0  # standard ballistic model

    return {
        "takeoff_time_s": float(t[takeoff_idx]),
        "landing_time_s": float(t[landing_idx]),
        "airtime_s": airtime,
        "height_m": height_m,
        "height_cm": height_m * 100.0,
        "height_inches": height_m * 39.3701,
    }


# ----------------------------
# 4. END-TO-END PIPELINE
# ----------------------------

def analyze_jumps_from_folder(folder_name: str, save_plots = False, result_dir = None) -> list[dict]:
    """
    High-level one-call API.

    Steps:
    - load accelerometer data
    - create magnitude-only dataframe
    - split into candidate jump windows around each landing
    - baseline-correct each window
    - detect takeoff & landing
    - compute airtime + jump height

    Returns
    -------
    results : list[dict]
        Each dict has:
            "takeoff_time_s"
            "landing_time_s"
            "airtime_s"
            "height_m"
            "height_cm"
            "window_df" (the processed per-jump dataframe with Magnitude_zeroed)
    """
    accel_df = load_accelerometer_folder(folder_name)
    try:
        mag_df = to_mag_df(accel_df)

    except Exception as e:
        raise ValueError(f"Error processing accelerometer data: {e}")

    jump_windows = split_into_jump_windows(mag_df)

    results = []
    for win_df in jump_windows:
        zeroed_df = zero_gravity(win_df)

        t_arr = zeroed_df["Time (s)"].to_numpy()
        a_arr = zeroed_df["Magnitude_zeroed"].to_numpy()

        takeoff_idx, landing_idx = detect_takeoff_v(t_arr, a_arr)

        metrics = compute_air_metrics(
            time_s=t_arr,
            takeoff_idx=takeoff_idx,
            landing_idx=landing_idx
        )

        plot_result = None
        if save_plots:
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
                plot_path = os.path.join(result_dir, f"jump_{len(results)+1}.png")
                plot_result = make_jump_graph(
                    zeroed_df,
                    takeoff_idx=takeoff_idx,
                    landing_idx=landing_idx,
                    title=f"Jump {len(results)+1}",
                    save_path=plot_path
                )
                with open(f"{result_dir}/jump_{len(results)+1}.json", "w") as f:
                    json.dump(metrics, f)

                
            else:
                plot_result = make_jump_graph(
                    zeroed_df,
                    takeoff_idx=takeoff_idx,
                    landing_idx=landing_idx
                )

        results.append({
            **metrics,
            "plot": plot_result,
            "window_df": zeroed_df  # keep this if you want to render / debug plot in UI
        })

    return results
