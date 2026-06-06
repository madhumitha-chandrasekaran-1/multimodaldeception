"""
03_extract_features.py
-----------------------
Extracts a comprehensive set of audio features from all WAV clips.

Feature families extracted:
  1. Prosodic      — pitch (F0) statistics, speech rate, pause patterns
  2. Spectral      — MFCCs, spectral centroid/rolloff/bandwidth/contrast, chroma
  3. Voice Quality — jitter, shimmer, HNR (via parselmouth/Praat)
  4. Energy        — RMS statistics, zero crossing rate
  5. Temporal      — voiced/unvoiced ratio, clip duration

Output:
    output/features/features.csv   — one row per clip, columns = features + label

Usage:
    python 03_extract_features.py --audio_dir output/audio/ --output_dir output/features/
"""

import argparse
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from typing import Tuple


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_audio(path: Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr


def safe_stat(arr: np.ndarray, stat: str) -> float:
    """Compute a statistic safely, returning 0.0 on failure."""
    arr = arr[~np.isnan(arr)] if arr is not None else np.array([])
    if len(arr) == 0:
        return 0.0
    try:
        return float({"mean": np.mean, "std": np.std, "median": np.median,
                      "max": np.max, "min": np.min, "range": lambda x: np.max(x) - np.min(x)}[stat](arr))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
# 1. Prosodic Features
# ──────────────────────────────────────────────

def extract_prosodic(y: np.ndarray, sr: int) -> dict:
    """
    Pitch (F0), speech rate proxy, pause patterns.
    """
    feats = {}

    # ── Pitch via pyin (more accurate than yin for speech) ──
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),   # ~65 Hz
        fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
        sr=sr
    )

    # Only use voiced frames
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

    feats["pitch_mean"]   = safe_stat(f0_voiced, "mean")
    feats["pitch_std"]    = safe_stat(f0_voiced, "std")
    feats["pitch_median"] = safe_stat(f0_voiced, "median")
    feats["pitch_range"]  = safe_stat(f0_voiced, "range")
    feats["pitch_max"]    = safe_stat(f0_voiced, "max")
    feats["pitch_min"]    = safe_stat(f0_voiced, "min")

    # Voiced frame ratio (proxy for continuous speech vs. pauses)
    feats["voiced_ratio"] = float(np.sum(voiced_flag) / len(voiced_flag)) if voiced_flag is not None else 0.0

    # ── Pause analysis via RMS energy ──
    frame_length = 512
    hop_length   = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    silence_threshold = 0.01
    is_silent = rms < silence_threshold
    feats["pause_ratio"] = float(np.mean(is_silent))

    # Count pause segments (runs of silence)
    pause_count = 0
    in_pause = False
    pause_lengths = []
    current_pause = 0

    for s in is_silent:
        if s and not in_pause:
            in_pause = True
            current_pause = 1
        elif s and in_pause:
            current_pause += 1
        elif not s and in_pause:
            pause_count += 1
            pause_lengths.append(current_pause)
            in_pause = False
            current_pause = 0

    frames_per_sec = sr / hop_length
    feats["pause_count"]      = float(pause_count)
    feats["pause_mean_dur"]   = float(np.mean(pause_lengths) / frames_per_sec) if pause_lengths else 0.0
    feats["pause_max_dur"]    = float(np.max(pause_lengths)  / frames_per_sec) if pause_lengths else 0.0

    # Duration
    feats["duration"] = float(librosa.get_duration(y=y, sr=sr))

    return feats


# ──────────────────────────────────────────────
# 2. Spectral Features
# ──────────────────────────────────────────────

def extract_spectral(y: np.ndarray, sr: int, n_mfcc: int = 20) -> dict:
    """
    MFCCs, spectral shape features, chroma.
    """
    feats = {}

    # ── MFCCs — mean and std of each coefficient ──
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfccs[i]))

    # ── Delta MFCCs (first-order derivative — captures dynamics) ──
    delta_mfcc = librosa.feature.delta(mfccs)
    for i in range(n_mfcc):
        feats[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        feats[f"delta_mfcc_{i+1}_std"]  = float(np.std(delta_mfcc[i]))

    # ── Spectral Centroid — "brightness" of sound ──
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    feats["spectral_centroid_std"]  = float(np.std(centroid))

    # ── Spectral Rolloff — frequency below which X% of energy lies ──
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["spectral_rolloff_mean"] = float(np.mean(rolloff))
    feats["spectral_rolloff_std"]  = float(np.std(rolloff))

    # ── Spectral Bandwidth ──
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    feats["spectral_bandwidth_std"]  = float(np.std(bandwidth))

    # ── Spectral Contrast — difference between peaks and valleys ──
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f"spectral_contrast_{i+1}_mean"] = float(np.mean(contrast[i]))

    # ── Zero Crossing Rate ──
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    # ── Chroma Features (12 pitch classes) ──
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))

    return feats


# ──────────────────────────────────────────────
# 3. Energy Features
# ──────────────────────────────────────────────

def extract_energy(y: np.ndarray, sr: int) -> dict:
    """RMS energy statistics."""
    feats = {}
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"]   = float(np.mean(rms))
    feats["rms_std"]    = float(np.std(rms))
    feats["rms_max"]    = float(np.max(rms))
    feats["rms_range"]  = float(np.max(rms) - np.min(rms))

    # Energy in first vs. second half of clip (does speaker start strong/weak?)
    mid = len(rms) // 2
    feats["rms_first_half"]  = float(np.mean(rms[:mid]))
    feats["rms_second_half"] = float(np.mean(rms[mid:]))
    feats["rms_trend"]       = feats["rms_second_half"] - feats["rms_first_half"]

    return feats


# ──────────────────────────────────────────────
# 4. Voice Quality Features (via Parselmouth/Praat)
# ──────────────────────────────────────────────

def extract_voice_quality(wav_path: Path) -> dict:
    """
    Jitter, shimmer, HNR — classical voice quality measures.
    Require Praat via parselmouth.

    These are particularly relevant for deception — stress and cognitive load
    cause involuntary micro-variations in vocal cord vibration.
    """
    feats = {}

    try:
        snd = parselmouth.Sound(str(wav_path))

        # ── Point process for jitter/shimmer ──
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # Jitter (local) — cycle-to-cycle pitch period variation
        feats["jitter_local"]    = call(point_process, "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_rap"]      = call(point_process, "Get jitter (rap)",              0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_ppq5"]     = call(point_process, "Get jitter (ppq5)",             0, 0, 0.0001, 0.02, 1.3)

        # Shimmer (local) — cycle-to-cycle amplitude variation
        feats["shimmer_local"]   = call([snd, point_process], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_apq3"]    = call([snd, point_process], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_apq5"]    = call([snd, point_process], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6)

        # HNR — Harmonics-to-Noise Ratio (higher = cleaner voice)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        feats["hnr_mean"] = call(harmonicity, "Get mean", 0, 0)
        feats["hnr_std"]  = call(harmonicity, "Get standard deviation", 0, 0)

    except Exception as e:
        print(f"    ⚠️  Voice quality extraction failed for {wav_path.name}: {e}")
        for key in ["jitter_local", "jitter_rap", "jitter_ppq5",
                    "shimmer_local", "shimmer_apq3", "shimmer_apq5",
                    "hnr_mean", "hnr_std"]:
            feats[key] = 0.0

    # Replace None/NaN with 0
    return {k: (0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
            for k, v in feats.items()}


# ──────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────

def extract_all_features(wav_path: Path, sr: int = 16000) -> dict:
    """Extract all feature families and merge into a single dict."""
    y, sr = load_audio(wav_path, sr)

    feats = {}
    feats.update(extract_prosodic(y, sr))
    feats.update(extract_spectral(y, sr))
    feats.update(extract_energy(y, sr))
    feats.update(extract_voice_quality(wav_path))

    return feats


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract audio features for deception detection")
    parser.add_argument("--audio_dir",  type=str, default="output/audio/",    help="Directory with truth/ and lie/ wav subfolders")
    parser.add_argument("--output_dir", type=str, default="output/features/", help="Where to save features.csv")
    parser.add_argument("--sr",         type=int, default=16000,               help="Sample rate")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_files = sorted((audio_dir / "truthful").glob("*.wav"))
    lie_files   = sorted((audio_dir / "deceptive").glob("*.wav"))
    all_files   = [(f, "truthful", 0) for f in truth_files] + [(f, "deceptive", 1) for f in lie_files]

    print("=" * 55)
    print("  Deception Detection — Feature Extraction")
    print("=" * 55)
    print(f"  Truth clips: {len(truth_files)}")
    print(f"  Lie clips:   {len(lie_files)}")
    print(f"  Total:       {len(all_files)}")
    print("=" * 55)

    records = []
    for i, (wav_path, label, label_int) in enumerate(all_files):
        print(f"  [{i+1:03d}/{len(all_files)}] {wav_path.name} ({label})")

        feats = extract_all_features(wav_path, args.sr)
        feats["filename"]  = wav_path.stem
        feats["label"]     = label
        feats["label_int"] = label_int
        records.append(feats)

    df = pd.DataFrame(records)

    # Move metadata cols to front
    meta_cols = ["filename", "label", "label_int"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    output_path = output_dir / "features.csv"
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 55)
    print(f"  ✅ Features saved: {output_path.resolve()}")
    print(f"  Clips:    {len(df)}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Classes:  {df['label'].value_counts().to_dict()}")
    print("=" * 55)


if __name__ == "__main__":
    main()