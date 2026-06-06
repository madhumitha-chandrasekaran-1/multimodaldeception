# Deception Detection — Audio Pipeline

Binary classification of **truth vs. deception** from short video clips using audio features.

## Project Structure

```
project/
├── data/
│   ├── truth/          ← .mp4 clips labelled truth
│   └── lie/            ← .mp4 clips labelled lie
├── output/
│   ├── audio/          ← extracted .wav files
│   ├── eda/            ← EDA plots and CSVs
│   ├── features/       ← extracted feature CSVs
│   └── results/        ← model results, plots, summary
├── 01_extract_audio.py
├── 02_eda.py
├── 03_extract_features.py
├── 04_train_evaluate.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

FFmpeg must be installed on your system:
- macOS:   `brew install ffmpeg`
- Ubuntu:  `sudo apt install ffmpeg`
- Windows: download from https://ffmpeg.org/download.html

## Usage — Run in Order

### Step 1: Extract Audio
```bash
python 01_extract_audio.py --data_dir data/ --output_dir output/audio/
```
Extracts mono 16kHz WAV from every .mp4 in data/truth/ and data/lie/

---

### Step 2: Exploratory Data Analysis
```bash
python 02_eda.py --audio_dir output/audio/ --output_dir output/eda/
```
Generates:
- Mel spectrograms (sample of truth + lie clips)
- Pitch contour overlays
- Energy contour overlays
- Feature distribution histograms (truth vs. lie)
- Summary statistics CSV

**Look at this before training any models.**

---

### Step 3: Extract Features
```bash
python 03_extract_features.py --audio_dir output/audio/ --output_dir output/features/
```
Extracts per-clip features across 4 families:
- **Prosodic**: pitch mean/std/range, pause ratio/count/duration, voiced ratio
- **Spectral**: 20 MFCCs + deltas, spectral centroid/rolloff/bandwidth/contrast, chroma, ZCR
- **Energy**: RMS stats, first/second half energy, trend
- **Voice Quality**: jitter (local/RAP/PPQ5), shimmer (local/APQ3/APQ5), HNR mean/std

Output: `output/features/features.csv` (~120 rows × ~140 features)

---

### Step 4: Train & Evaluate
```bash
python 04_train_evaluate.py \
    --features_path output/features/features.csv \
    --audio_dir output/audio/ \
    --output_dir output/results/
```
Runs 3 experiments with stratified 10-fold CV:

| Experiment | Features | Models |
|---|---|---|
| 1 | Hand-crafted (librosa + parselmouth) | SVM, RF, XGBoost, LR |
| 2 | Mel Spectrogram → Frozen ResNet18 | SVM, RF, XGBoost, LR |
| 3 | Fusion (1 + 2) | SVM, RF, XGBoost, LR |

Generates:
- `cv_results.csv` — all metrics
- `results_summary.txt` — human-readable best model summary
- `results_comparison.png` — bar chart comparison
- `confusion_*.png` — confusion matrices per experiment
- `roc_*.png` — ROC curves per experiment
- `feature_importance.png` — top 25 features by permutation importance

#### Skip CNN experiment (if PyTorch not available):
```bash
python 04_train_evaluate.py --skip_cnn
```

---

## Key Design Decisions

**Why 10-fold CV?** With 120 clips, a single train/test split wastes data and gives unreliable estimates. 10-fold CV uses all data for evaluation.

**Why global z-score normalization?** No speaker metadata is available, so per-speaker normalization isn't possible. The speaker identity confound is a known limitation.

**Why frozen ResNet18?** 120 clips is too small to fine-tune a CNN. Using it as a frozen feature extractor gives us the representational power without overfitting.

**Why permutation importance?** Model-agnostic and works with pipelines containing a scaler. Shows which features actually matter for prediction.

---

## Limitations

- **Speaker confound**: Different speakers for truth vs. lie means the model may partly learn speaker identity rather than deception patterns.
- **Small dataset**: 120 clips limits generalization. Results should be interpreted with caution.
- **No temporal modeling**: Features are summary statistics per clip; an LSTM/RNN could capture dynamics better with more data.
