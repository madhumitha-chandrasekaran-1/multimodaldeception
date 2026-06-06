"""
02_eda.py
----------
Exploratory Data Analysis on extracted audio clips.
Generates visual comparisons of truth vs. lie across:
  - Mel spectrograms
  - Pitch (F0) contours
  - Energy contours
  - Feature distributions (pitch, speech rate, pause ratio, MFCCs)

Usage:
    python 02_eda.py --audio_dir output/audio/ --output_dir output/eda/

Output:
    output/eda/
        spectrograms_truth.png
        spectrograms_lie.png
        pitch_contours.png
        energy_contours.png
        feature_distributions.png
        summary_stats.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
from pathlib import Path
from typing import Tuple, List


# ──────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────

def load_audio(path: Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


# ──────────────────────────────────────────────
# Basic feature extraction for EDA
# ──────────────────────────────────────────────

def extract_eda_features(y: np.ndarray, sr: int) -> dict:
    """Extract a small set of interpretable features for EDA."""

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Pitch (F0) via pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    pitch_mean  = float(np.nanmean(f0)) if len(f0_voiced) > 0 else 0.0
    pitch_std   = float(np.nanstd(f0))  if len(f0_voiced) > 0 else 0.0
    pitch_range = float(np.nanmax(f0) - np.nanmin(f0)) if len(f0_voiced) > 0 else 0.0

    # Energy (RMS)
    rms         = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # Speech rate proxy: zero crossing rate
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    # Pause ratio: fraction of frames with RMS below threshold
    silence_threshold = 0.01
    pause_ratio = float(np.mean(rms < silence_threshold))

    # MFCCs (mean of first 13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = {f"mfcc_{i+1}_mean": float(np.mean(mfccs[i])) for i in range(13)}

    return {
        "duration":     duration,
        "pitch_mean":   pitch_mean,
        "pitch_std":    pitch_std,
        "pitch_range":  pitch_range,
        "energy_mean":  energy_mean,
        "energy_std":   energy_std,
        "zcr_mean":     zcr_mean,
        "pause_ratio":  pause_ratio,
        **mfcc_means
    }


# ──────────────────────────────────────────────
# Plot: Sample spectrograms
# ──────────────────────────────────────────────

def plot_spectrograms(files: List[Path], label: str, output_path: Path, n: int = 6, sr: int = 16000):
    """Plot mel spectrograms for a sample of clips."""
    files_sample = files[:n]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Mel Spectrograms — {label.upper()} clips (sample of {n})", fontsize=14, fontweight='bold')

    for i, (ax, fpath) in enumerate(zip(axes.flatten(), files_sample)):
        y, sr_ = load_audio(fpath, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr_, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr_, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title(fpath.stem, fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel freq")
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


# ──────────────────────────────────────────────
# Plot: Pitch contours overlay
# ──────────────────────────────────────────────

def plot_pitch_contours(truth_files: List[Path], lie_files: List[Path], output_path: Path, n: int = 5, sr: int = 16000):
    """Overlay pitch contours for truth vs lie clips."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Pitch (F0) Contours: Truth vs. Lie", fontsize=14, fontweight='bold')

    for ax, files, label, color in zip(
        axes,
        [truth_files[:n], lie_files[:n]],
        ["TRUTH", "LIE"],
        ["steelblue", "tomato"]
    ):
        for fpath in files:
            y, sr_ = load_audio(fpath, sr=sr)
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), sr=sr_
            )
            times = librosa.times_like(f0, sr=sr_)
            ax.plot(times, f0, alpha=0.5, color=color, linewidth=1.2, label=fpath.stem)

        ax.set_title(f"{label} clips (n={n})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pitch Hz")
        ax.set_ylim(0, 400)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


# ──────────────────────────────────────────────
# Plot: Feature distributions
# ──────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, output_path: Path):
    """Plot distributions of key features by label."""
    features = ["duration", "pitch_mean", "pitch_std", "pitch_range",
                "energy_mean", "pause_ratio", "zcr_mean",
                "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean"]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Feature Distributions: Truth vs. Lie", fontsize=14, fontweight='bold')

    for ax, feat in zip(axes.flatten(), features):
        truth_vals = df[df["label"] == "truthful"][feat].dropna()
        lie_vals   = df[df["label"] == "deceptive"][feat].dropna()

        ax.hist(truth_vals, bins=15, alpha=0.6, color="steelblue", label="Truth", density=True)
        ax.hist(lie_vals,   bins=15, alpha=0.6, color="tomato",    label="Lie",   density=True)

        # Add mean lines
        ax.axvline(truth_vals.mean(), color="steelblue", linestyle="--", linewidth=1.5)
        ax.axvline(lie_vals.mean(),   color="tomato",    linestyle="--", linewidth=1.5)

        ax.set_title(feat.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


# ──────────────────────────────────────────────
# Plot: Energy contours
# ──────────────────────────────────────────────

def plot_energy_contours(truth_files: List[Path], lie_files: List[Path], output_path: Path, n: int = 5, sr: int = 16000):
    """Plot RMS energy over time for truth vs lie."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("RMS Energy Contours: Truth vs. Lie", fontsize=14, fontweight='bold')

    for ax, files, label, color in zip(
        axes,
        [truth_files[:n], lie_files[:n]],
        ["TRUTH", "LIE"],
        ["steelblue", "tomato"]
    ):
        for fpath in files:
            y, sr_ = load_audio(fpath, sr=sr)
            rms   = librosa.feature.rms(y=y)[0]
            times = librosa.times_like(rms, sr=sr_)
            ax.plot(times, rms, alpha=0.5, color=color, linewidth=1.2)

        ax.set_title(f"{label} clips (n={n})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS Energy")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA for deception detection audio clips")
    parser.add_argument("--audio_dir",  type=str, default="output/audio/", help="Directory with truth/ and lie/ wav folders")
    parser.add_argument("--output_dir", type=str, default="output/eda/",   help="Where to save EDA plots and CSVs")
    parser.add_argument("--sr",         type=int, default=16000,            help="Sample rate")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_dir = audio_dir / "truthful"
    lie_dir   = audio_dir / "deceptive"

    truth_dir = audio_dir / "truthful"
    lie_dir   = audio_dir / "deceptive"

    truth_files = sorted(truth_dir.glob("*.wav"))
    lie_files   = sorted(lie_dir.glob("*.wav"))

    print("=" * 55)
    print("  Deception Detection — EDA")
    print("=" * 55)
    print(f"  Truth clips: {len(truth_files)}")
    print(f"  Lie clips:   {len(lie_files)}")
    print("=" * 55)

    # ── 1. Spectrograms ──
    print("\n📊 Generating spectrograms...")
    plot_spectrograms(truth_files, "truth", output_dir / "spectrograms_truth.png")
    plot_spectrograms(lie_files,   "lie",   output_dir / "spectrograms_lie.png")

    # ── 2. Pitch contours ──
    print("\n🎵 Generating pitch contours...")
    plot_pitch_contours(truth_files, lie_files, output_dir / "pitch_contours.png")

    # ── 3. Energy contours ──
    print("\n⚡ Generating energy contours...")
    plot_energy_contours(truth_files, lie_files, output_dir / "energy_contours.png")

    # ── 4. Feature distributions ──
    print("\n📈 Extracting features for distribution plots...")
    records = []
    all_files = [(f, "truth") for f in truth_files] + [(f, "lie") for f in lie_files]

    for fpath, label in all_files:
        print(f"  Processing {fpath.name}...")
        y, sr = load_audio(fpath, args.sr)
        feats = extract_eda_features(y, sr)
        feats["filename"] = fpath.stem
        feats["label"]    = label
        records.append(feats)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "eda_features.csv", index=False)
    print(f"  ✅ Saved: eda_features.csv ({len(df)} clips, {len(df.columns)} columns)")

    plot_feature_distributions(df, output_dir / "feature_distributions.png")

    # ── 5. Summary stats ──
    print("\n📋 Summary statistics by label:")
    summary = df.groupby("label")[["duration", "pitch_mean", "pitch_std",
                                    "energy_mean", "pause_ratio", "zcr_mean"]].agg(["mean", "std"])
    print(summary.to_string())
    summary.to_csv(output_dir / "summary_stats.csv")
    print(f"\n  ✅ Saved: summary_stats.csv")

    print("\n" + "=" * 55)
    print(f"  🎉 EDA complete! Results saved to: {output_dir.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()