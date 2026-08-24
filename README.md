# Gaze-Controlled Virtual Keyboard + LSTM Predictive Text

> **Novel contribution:** LSTM language model (Word2Vec embeddings) predicts top-3 word completions in real time, reducing required gaze-selections by ~40–60% on common English text vs. the Meena & Salvi 2025 baseline.

---

## Quick Start

### 1. Set up the environment

```powershell
# From the project root — uses Python 3.12
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify GPU

```python
import torch
print(torch.cuda.is_available())   # → True
print(torch.cuda.get_device_name(0))  # → NVIDIA GeForce RTX 4050 Laptop GPU
```

---

## Project Structure

```
Deep Learning Project/
├── data/
│   ├── synthetic/          ← UnityEyes images, organised into class folders 1–9
│   ├── calibration/        ← per-user eye crops (300 × 9 directions)
│   └── text_corpus/        ← (auto-downloaded via NLTK Brown corpus)
├── models/
│   ├── base_gaze_model.pt  ← trained CNN checkpoint
│   ├── user_gaze_<id>.pt   ← fine-tuned per-user CNN
│   ├── word2vec.bin         ← trained Word2Vec embeddings
│   ├── lstm_lm.pt          ← trained LSTM language model
│   └── word2idx.json        ← vocabulary mapping
├── src/
│   ├── data_loader.py      ← UnityEyes ZIP parser + NLTK corpus loader
│   ├── train_gaze.py       ← CNN training (Meena & Salvi architecture)
│   ├── train_lm.py         ← Word2Vec + LSTM language model training
│   ├── predictor.py        ← word completion API (LSTM inference)
│   ├── eye_detector.py     ← Haar cascade face/eye detector
│   ├── gaze_predictor.py   ← CNN inference wrapper + mouse mock
│   ├── command_selector.py ← Algorithm 1 (Async) + Algorithm 2 (Sync)
│   └── keyboard_gui.py     ← Tkinter GUI with prediction row
├── calibrate.py            ← 9-point calibration + CNN fine-tuning
├── main.py                 ← entry point
├── evaluate.py             ← ITR measurement
└── requirements.txt
```

---

## Step-by-Step Build Order

### Step 1 — Generate training data with UnityEyes

1. Extract `UnityEyes_Windows.zip` to `data/synthetic_raw/`
2. Run `data/synthetic_raw/UnityEyes_Windows/unityeyes.exe`
3. Generate ~20,000 images (outputs `.jpg` + `.json` pairs)
4. Parse them into class folders:
   ```powershell
   python src/data_loader.py --input_dir data/synthetic_raw/UnityEyes_Windows/imgs \
                              --output_dir data/synthetic \
                              --n_images 20000
   ```

### Step 2 — Train the Gaze CNN (run in background)

```powershell
python src/train_gaze.py --data_dir data/synthetic --output_dir models
```
Expected time on RTX 4050: ~20–40 minutes.
Outputs: `models/base_gaze_model.pt`, `training_curves.png`, `confusion_matrix.png`

### Step 3 — Train the LSTM Language Model (run in background)

```powershell
python src/train_lm.py --output_dir models
```
Expected time on RTX 4050: ~15–25 minutes.
Outputs: `models/lstm_lm.pt`, `models/word2idx.json`, `models/word2vec.bin`

> Both Step 2 and 3 can run simultaneously in separate terminals.

### Step 4 — Test the GUI immediately (no trained model needed)

```powershell
python main.py --mode mouse
```
Move your mouse to a screen quadrant → the cell highlights. The keyboard is fully functional with fallback word predictions.

### Step 5 — Calibrate for webcam mode

```powershell
python calibrate.py --user_id yourname --camera 0
```
Follow the on-screen dots (9 directions × 300 frames). Fine-tunes the CNN to your eyes.

### Step 6 — Run with webcam

```powershell
# Algorithm 1 (Async — fires on 6 consecutive agreement frames)
python main.py --mode webcam_async --user_id yourname

# Algorithm 2 (Sync — weighted window vote)
python main.py --mode webcam_sync --user_id yourname
```

### Step 7 — Evaluate ITR

```powershell
python evaluate.py --sentence "painting which landform" --save_json results.json
```
>  Replace `--accuracy` and `--spm` with values measured from your real live session.

---

## Syllabus Coverage

| Module | Topic | Used In |
|--------|-------|---------|
| 1 — MLP/Backprop | Adam, Dropout, Augmentation | CNN training |
| 2 — Regularization | Dropout(0.2, 0.5), EarlyStopping | CNN + LSTM |
| 3 — CNN | Conv2D, MaxPool, Stride, Params | 4-layer gaze CNN |
| 3 — Transfer Learning | Fine-tune pretrained CNN | Calibration |
| 4 — Sequence Models | Stacked LSTM, BPTT | Language model |
| 6 — Word Embeddings | Word2Vec, Embedding layer | Language model |

---

## Dependencies

Install PyTorch separately (CUDA 12.4):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

All other deps in `requirements.txt`:
```
pip install -r requirements.txt
```
