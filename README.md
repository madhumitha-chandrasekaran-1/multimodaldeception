# Multimodal Deception Detection — Audio Pipeline

Binary classification of **truthful vs. deceptive speech** from short video clips using hand-crafted audio features and classical ML.

---

## What It Does

The project takes video clips of people speaking (labeled truthful or deceptive) and predicts whether a speaker is lying — purely from their voice. No visual information is used. The pipeline extracts acoustic cues that humans unconsciously produce when under the cognitive load of deception: elevated pitch, irregular vocal cord vibration, more frequent pauses, and shifts in spectral energy distribution.

---

## Dataset Structure

```
data/
├── truthful/     ← .mp4 clips of truthful speech
└── deceptive/    ← .mp4 clips of deceptive speech
```

The pipeline expects MP4 video clips organized into these two folders. Audio is extracted automatically.

---

## Pipeline — 4 Steps

```
01_extract_audio.py   →   02_eda.py   →   03_extract_features.py   →   04_train_evaluate.py
```

### Step 1 — Audio Extraction
```bash
python 01_extract_audio.py --data_dir data/ --output_dir output/audio/
```
Uses **FFmpeg** to strip audio from every `.mp4` and convert it to **16kHz mono WAV** — the standard format for speech processing. Output lands in `output/audio/truthful/` and `output/audio/deceptive/`.

---

### Step 2 — Exploratory Data Analysis
```bash
python 02_eda.py --audio_dir output/audio/ --output_dir output/eda/
```
Generates visual comparisons of truthful vs. deceptive clips:
- Mel spectrograms (sample clips per class)
- Pitch (F0) contour overlays
- RMS energy contour overlays
- Feature distribution histograms (pitch, pause ratio, MFCCs, energy, ZCR)
- `summary_stats.csv` — mean/std per class for key features

---

### Step 3 — Feature Extraction
```bash
python 03_extract_features.py --audio_dir output/audio/ --output_dir output/features/
```

Extracts ~140 features per clip across four families:

| Family | Features |
|---|---|
| **Prosodic** | Pitch mean/std/median/range/max/min, voiced ratio, pause ratio/count/mean duration/max duration, clip duration |
| **Spectral** | 20 MFCCs + 20 delta-MFCCs (mean & std each), spectral centroid/rolloff/bandwidth/contrast, chroma (12 bins), ZCR |
| **Energy** | RMS mean/std/max/range, first-half vs. second-half RMS, energy trend |
| **Voice Quality** | Jitter (local, RAP, PPQ5), Shimmer (local, APQ3, APQ5), HNR mean/std — via **Praat/parselmouth** |

Output: `output/features/features.csv` — one row per clip, one column per feature.

Voice quality features (jitter, shimmer, HNR) are the most forensically grounded: stress and cognitive load cause involuntary micro-variations in vocal cord vibration that are nearly impossible to fake.

---

### Step 4 — Training & Evaluation
```bash
python 04_train_evaluate.py \
    --features_path output/features/features.csv \
    --output_dir output/results/
```

Trains four classifiers with **5-fold stratified cross-validation** and `StandardScaler` normalization:

| Model | Notes |
|---|---|
| Logistic Regression | L2 regularization, C=0.1 |
| SVM (RBF kernel) | C=2.0, tuned for this dataset |
| Extra Trees | 100 estimators, max_depth=5 — handles noisy high-dim audio features well |
| Ensemble (LR + SVM + ET) | Soft-voting average of all three |

---

## Results

| Model | Accuracy | AUC | F1 |
|---|---|---|---|
| **SVM (RBF)** | **0.768 ± 0.058** | **0.851 ± 0.090** | **0.762 ± 0.072** |
| Ensemble (LR+SVM+ET) | 0.759 ± 0.082 | 0.840 ± 0.087 | 0.743 ± 0.100 |
| Extra Trees | 0.726 ± 0.065 | 0.821 ± 0.058 | 0.717 ± 0.067 |
| Logistic Regression | 0.743 ± 0.069 | 0.807 ± 0.091 | 0.747 ± 0.062 |

The SVM with an RBF kernel achieves the best AUC (0.851), well above chance (0.5). This means the model correctly ranks a deceptive speaker above a truthful one 85% of the time.

### Output Files

```
output/results/
├── tuned_cv_results.csv       ← all CV metrics per model
├── results_summary.txt        ← human-readable summary
├── results_comparison.png     ← bar chart: accuracy / AUC / F1
├── confusion_tuned.png        ← confusion matrices for all models
├── roc_tuned.png              ← ROC curves overlay
└── feature_importance.png     ← top 20 features by permutation importance
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Audio extraction | FFmpeg |
| Audio analysis | librosa |
| Voice quality (Praat) | parselmouth |
| ML models | scikit-learn (SVM, ExtraTrees, LogisticRegression, VotingClassifier) |
| Data | numpy, pandas |
| Visualization | matplotlib |

---

## Where This Is Useful

- **Forensic analysis** — screening interview recordings for potential deception
- **Security & intelligence** — supporting (not replacing) human judgment in interrogation contexts
- **Academic research** — studying vocal correlates of deception and cognitive load
- **Automated interview systems** — flagging responses for human review

This is a decision-support tool, not an autonomous lie detector. Deception detection from audio alone is inherently probabilistic and should be treated as one signal among many.

---

## Setup

```bash
pip install -r requirements.txt
```

FFmpeg must be installed separately:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## Project Structure

```
.
├── 01_extract_audio.py        ← video → 16kHz mono WAV
├── 02_eda.py                  ← spectrograms, pitch/energy plots
├── 03_extract_features.py     ← ~140 acoustic features per clip
├── 04_train_evaluate.py       ← cross-validation, plots, summary
├── requirements.txt
├── results_summary.txt        ← final model results
├── tuned_cv_results.csv       ← raw CV metrics
├── confusion_tuned.png
├── roc_tuned.png
├── feature_importance.png
└── results_comparison.png
```

---

## Limitations

- **Speaker confound**: Different speakers across truth/lie classes means the model may partly learn speaker identity rather than deception patterns. Per-speaker normalization isn't possible without speaker metadata.
- **Small dataset**: The dataset size limits generalization — results should be interpreted with caution.
- **Audio only**: No visual cues (facial expressions, gesture, gaze) are used. A true multimodal system would likely outperform audio-only.
- **No temporal modeling**: Features are summary statistics per clip. An LSTM or Transformer could capture fine-grained temporal dynamics with more data.
- **Context-free**: The model has no access to ground-truth transcripts, topic, or question type — all of which affect baseline pitch and speech rate in honest speech.
