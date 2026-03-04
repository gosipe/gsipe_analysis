"""Read CSV, extract 2nd column, and apply 10-100 Hz bandpass filter.

Usage:
  - Set `CSV_PATH` and `FS` at top of file.
  - Run: `python scripts/lick_bandpass_filter.py`

Outputs the filtered signal as a NumPy .npy file next to the CSV.
"""
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# --- User-configurable variables ---
# Path to input CSV file. Update this to your file path.
CSV_PATH = r"C://Users//Graybird//Desktop//week_5_lick_detection.csv"
# Sampling frequency (Hz) of the data in the CSV. Update as needed.
FS = 30.0


def load_second_column(csv_path, col_index=14, delimiter=',', skip_header=False):
    """Load the second column (index 1) from a CSV file as a 1D numpy array.

    Attempts a robust load: if the file has a header, set `skip_header=True`.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        if skip_header:
            data = np.loadtxt(p, delimiter=delimiter, skiprows=1, usecols=(col_index,))
        else:
            data = np.loadtxt(p, delimiter=delimiter, usecols=(col_index,))
    except ValueError:
        # Fall back to genfromtxt for mixed types or missing values
        data = np.genfromtxt(p, delimiter=delimiter, skip_header=1 if skip_header else 0, usecols=(col_index,))

    return np.asarray(data, dtype=float)


def bandpass_filter(data, fs, lowcut=10.0, highcut=14.0, order=4):
    """Apply a Butterworth bandpass filter (zero-phase) to 1D data.

    Args:
        data: 1D numpy array
        fs: sampling frequency in Hz
        lowcut: low cutoff frequency in Hz
        highcut: high cutoff frequency in Hz
        order: filter order

    Returns:
        filtered 1D numpy array
    """
    nyq = 0.5 * fs
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be > 0")
    if lowcut >= highcut:
        raise ValueError("lowcut must be < highcut")
    if highcut >= nyq:
        raise ValueError(f"highcut must be < Nyquist ({nyq} Hz)")

    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Use filtfilt for zero-phase filtering
    filtered = filtfilt(b, a, data)
    return filtered


def main():
    csv_path = CSV_PATH
    fs = FS

    # Try to detect if the CSV has a header by reading the first line
    header_flag = False
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            first = f.readline()
            # simple heuristic: if first line contains non-numeric characters besides separators
            if any(c.isalpha() for c in first):
                header_flag = True
    except FileNotFoundError:
        raise

    # Load the 2nd column (index 1)
    sig = load_second_column(csv_path, col_index=1, skip_header=header_flag)

    # Basic sanity checks
    if sig.ndim != 1:
        sig = sig.ravel()

    # Apply bandpass filter 10-100 Hz (handle bad sampling rates)
    try:
        filtered = bandpass_filter(sig, fs, lowcut=10.0, highcut=14.0, order=4)
    except ValueError as e:
        print(f"Warning: could not apply 10-100 Hz bandpass: {e}")
        print("Proceeding with unfiltered signal for plotting/saving.")
        filtered = sig.copy()

    # Save filtered output next to CSV
    out_path = Path(csv_path).with_suffix('.filtered.npy')
    np.save(out_path, filtered)
    print(f"Filtered signal saved to: {out_path}")

    # Plot original and filtered time series
    t = np.arange(sig.size) / float(fs)
    plt.figure(figsize=(10, 4))
    plt.plot(t, sig, label='raw', alpha=0.6)
    plt.plot(t, filtered, label='filtered', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Raw vs Filtered (10-100 Hz)')
    plt.legend()
    plt.tight_layout()
    fig_path = Path(csv_path).with_suffix('.filtered.png')
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
