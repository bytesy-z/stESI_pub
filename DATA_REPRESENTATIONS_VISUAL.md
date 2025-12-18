# stESI Data Representations - Visual Summary

## 📊 All Data Representations Used in the Project

### Legend
- 📐 = Geometric/Anatomical Data
- 📈 = Signal/Time-Series Data
- 🧠 = Machine Learning Data
- 📦 = Container/Collection Data
- 🎯 = Results/Output Data
- 📋 = Configuration/Metadata
- 🎨 = Visualization Data

---

## 1. ANATOMICAL DATA 📐

```
┌─────────────────────────────────────────────────────────────┐
│ HeadModel (Brain Anatomy)                                   │
├─────────────────────────────────────────────────────────────┤
│ • Subject: fsaverage                                        │
│ • Contains:                                                  │
│   - ElectrodeSpace: 59 electrodes (standard_1020)           │
│     └─ positions (59 × 3) meters, fs=500 Hz                │
│   - SourceSpace: 1289 sources (ico3 sampling)              │
│     └─ positions (1289 × 3) m, orientations (1289 × 3)    │
│   - ForwardModel: Leadfield (59 × 1289) V/A                │
│     └─ BEM model with conductivity (0.3, 0.006, 0.3)      │
└─────────────────────────────────────────────────────────────┘
```

**Data Type**: HeadModel class object
**File Format**: .fif (MNE forward), .mat (leadfield, positions)
**Dimensions**: Leadfield (n_electrodes × n_sources)
**SI Units**: meters (positions), V/A (leadfield)

---

## 2. SIGNAL DATA 📈

### 2.1 Raw EEG Signal
```
┌─────────────────────────────────────────────────────────────┐
│ EEG Signal (Raw Brain Recordings)                           │
├─────────────────────────────────────────────────────────────┤
│ Shape: (n_electrodes=59, n_times=500)                      │
│ Duration: 1 second (@ 500 Hz)                               │
│ Range: ±100 μV typical                                      │
│ Data Type: float32/float64                                  │
│ File Format: EDF / MAT / NPZ                                │
│ SNR: 5-25 dB (configurable)                                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Source Activity
```
┌─────────────────────────────────────────────────────────────┐
│ Source Activity (Neural Current Densities)                  │
├─────────────────────────────────────────────────────────────┤
│ Shape: (n_sources=1289, n_times=500)                       │
│ Units: A⋅m (Ampere-meters)                                 │
│ Range: ±1e-8 A⋅m typical                                   │
│ Orientation: Constrained (1D) or Unconstrained (3D)        │
│ Data Type: float32/float64                                  │
│ File Format: MAT                                             │
│ Metadata: active_source_indices, seed_indices              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Normalization Parameters
```
┌─────────────────────────────────────────────────────────────┐
│ Normalization Scheme                                         │
├─────────────────────────────────────────────────────────────┤
│ Method 1: Max-Max Normalization                             │
│   Output range: [0, 1]                                       │
│   Formula: (x - min) / (max - min)                          │
│                                                              │
│ Method 2: Linear Normalization                              │
│   Output range: [-scale, scale]                             │
│   Formula: x / max_abs                                      │
│                                                              │
│ Method 3: Global 99th Percentile                            │
│   Uses global norm instead of per-window                    │
│   Reduces artificial amplitude variations                   │
│                                                              │
│ Storage: NormalizationParams                                │
│   • max_eeg: float                                          │
│   • max_src: float                                          │
│   • scale_factor: float                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. DATASET & BATCH DATA 📦

### 3.1 Dataset
```
┌─────────────────────────────────────────────────────────────┐
│ EsiDatasetds_new (Training/Validation Dataset)              │
├─────────────────────────────────────────────────────────────┤
│ Type: SEREEGA or NMM (Neural Mass Model)                    │
│ Size: N samples (configurable, e.g., 1000)                 │
│                                                              │
│ Per Sample:                                                  │
│   • EEG signal: (59, 500) μV                               │
│   • Source activity: (1289, 500) A⋅m                       │
│   • Metadata: active sources, seeds                         │
│                                                              │
│ Attributes:                                                  │
│   • ids: [0, 1, 2, ..., N-1]                               │
│   • eeg_dict: {id → file_path}                             │
│   • src_dict: {id → file_path}                             │
│   • match_dict: {id → metadata_path}                       │
│   • md_dict: {id → metadata_json}                          │
│   • max_eeg, max_src: normalization values                 │
│   • snr_db: signal-to-noise ratio                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Training Batch
```
┌─────────────────────────────────────────────────────────────┐
│ TrainingBatch (Mini-batch for Neural Network)               │
├─────────────────────────────────────────────────────────────┤
│ Input Tensor (eeg_signals):                                 │
│   Shape: (batch_size=8, n_electrodes=59, n_times=500)     │
│   Type: torch.Tensor (float32)                              │
│   Range: [0, 1] (normalized)                                │
│                                                              │
│ Output Tensor (source_activities):                          │
│   Shape: (batch_size=8, n_sources=1289, n_times=500)      │
│   Type: torch.Tensor (float32)                              │
│   Range: [0, 1] (normalized)                                │
│                                                              │
│ Associated Data:                                             │
│   • max_value_eeg: Tensor for denormalization              │
│   • max_value_src: Tensor for denormalization              │
│   • normalization: "max-max" | "linear"                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. NEURAL NETWORK MODEL 🧠

### 4.1 Model Architecture
```
┌─────────────────────────────────────────────────────────────┐
│ CNN1Dpl (PyTorch Lightning Model)                           │
├─────────────────────────────────────────────────────────────┤
│ Architecture: simple_1dCNN_v2                               │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Input: (batch, 59, 500) - EEG signals              │   │
│ │         ↓                                            │   │
│ │ Conv1D: in_channels=59 → out_channels=4096         │   │
│ │         kernel_size=5, dilation=1, padding='same'  │   │
│ │         ↓                                            │   │
│ │ ReLU activation                                      │   │
│ │ Transpose: (batch, 4096, 500) → (batch, 500, 4096)│   │
│ │         ↓                                            │   │
│ │ Linear: 4096 → 1289 (per time step)                │   │
│ │         ↓                                            │   │
│ │ Output: (batch, 1289, 500) - Source predictions    │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ Training Details:                                            │
│   • Loss: cosine similarity, MSE, or logMSE               │
│   • Optimizer: Adam (lr=1e-3)                             │
│   • Batch size: 8                                          │
│   • Gradient clipping: 1.0 for LSTM                       │
│   • Early stopping: patience=20                           │
│   • Epochs: 100 typical                                    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Alternative Models
```
LSTM (Recurrent)
  • HeckerLSTMpl class
  • Temporal modeling
  
DeepSIF
  • Spatial-temporal fusion
  • num_sensor × temporal_input_size → num_source
```

---

## 5. INFERENCE RESULTS 🎯

### 5.1 Inference Processing
```
┌─────────────────────────────────────────────────────────────┐
│ Inference Pipeline: EDF File → Animation Data               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ STEP 1: Segmentation                                        │
│   • Input EEG: (n_electrodes, long_duration)               │
│   • Window size: 500 samples (1 sec)                        │
│   • Overlap: 50% → 250 sample step                         │
│   • Output: N windows of (59, 500)                         │
│                                                              │
│ STEP 2: Per-Window Inference                                │
│   for each window:                                          │
│     • Normalize using NormalizationParams                   │
│     • Pass through CNN1D model                              │
│     • Get predictions: (1289, 500)                         │
│     • Store as InferenceResult                             │
│                                                              │
│ STEP 3: Aggregation                                         │
│   • Collect all window predictions                          │
│   • Optionally compare with ground truth                    │
│   • Compute MetricsResult (NMSE, AUC, errors)             │
│                                                              │
│ STEP 4: Temporal Smoothing (Optional)                       │
│   • Apply EMA smoothing with α=0.3-0.7                    │
│   • Bidirectional (forward + backward passes)             │
│   • Warning: Heavy smoothing degrades accuracy ~80%       │
│                                                              │
│ STEP 5: Animation Timeline Generation                       │
│   • Stack all predictions: (1289, N_windows)              │
│   • Create timestamps: (N_windows)                         │
│   • Package into AnimationData                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Single Inference Result
```
┌─────────────────────────────────────────────────────────────┐
│ InferenceResult (Per Window)                                │
├─────────────────────────────────────────────────────────────┤
│ Window Index: 5                                              │
│ Source Predictions: (1289, 500) array - A⋅m               │
│ Timestamp: 2.5 seconds                                      │
│ Processing Time: 45 ms                                      │
│ Confidence: 0.85 (if available)                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Evaluation Metrics
```
┌─────────────────────────────────────────────────────────────┐
│ MetricsResult (Ground Truth Comparison)                     │
├─────────────────────────────────────────────────────────────┤
│ NMSE (Normalized Mean Squared Error): 0.12                 │
│ AUC (Area Under ROC Curve): 0.92                           │
│ Localization Error: 8.5 mm                                  │
│ Time Error: 45 ms                                           │
│ Seed Indices (GT): [123, 456, 789]                         │
│ Estimated Seeds: [125, 455, 792]                           │
│ Peak Times (GT): [150, 250, 350] samples                   │
│ Peak Times (Pred): [152, 248, 352] samples                 │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Segment Summary
```
┌─────────────────────────────────────────────────────────────┐
│ SegmentSummary (Window Metadata)                            │
├─────────────────────────────────────────────────────────────┤
│ Window Index: 5                                              │
│ Start Sample: 1250                                          │
│ Start Time: 2.5 seconds                                     │
│ EEG Max Amplitude: 42.3 μV                                 │
│ Output File: /results/edf_inference/.../segment_5.mat     │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. VISUALIZATION DATA 🎨

### 6.1 Animation Timeline
```
┌─────────────────────────────────────────────────────────────┐
│ AnimationTimeline (Source Activity Over Time)               │
├─────────────────────────────────────────────────────────────┤
│ Activity Matrix:                                             │
│   Shape: (1289 sources, 40 windows)                        │
│   Range: [0, 1] (normalized)                                │
│   Units: Normalized activity                                │
│                                                              │
│ Time Points:                                                 │
│   Shape: (40,)                                              │
│   Values: [0.0, 0.5, 1.0, 1.5, ..., 19.5] seconds         │
│                                                              │
│ Smoothing: α = 0.3                                          │
│   Forward + backward EMA pass                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Animation Data (NPZ Archive)
```
┌─────────────────────────────────────────────────────────────┐
│ AnimationData (Complete 3D Visualization Package)            │
├─────────────────────────────────────────────────────────────┤
│ File Format: NumPy .npz (compressed)                         │
│                                                              │
│ Contents:                                                    │
│   1. activity_timeline: (1289, 40) float32                  │
│      └─ Source activation per window                        │
│                                                              │
│   2. timestamps: (40,) float32                              │
│      └─ Time points in seconds                              │
│                                                              │
│   3. brain_vertices: (10240, 3) float32                    │
│      └─ Brain mesh vertex coordinates (meters)             │
│                                                              │
│   4. brain_faces: (20480, 3) int32                          │
│      └─ Triangle face indices for mesh                      │
│                                                              │
│ Usage:                                                       │
│   • Load in browser/3D viewer                               │
│   • Animate mesh colors by activity_timeline               │
│   • Play timeline from 0 to max timestamps                  │
│                                                              │
│ File Size: ~2-5 MB typical                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. CONFIGURATION & METADATA 📋

### 7.1 Simulation Configuration
```
┌─────────────────────────────────────────────────────────────┐
│ Configuration (JSON)                                         │
├─────────────────────────────────────────────────────────────┤
│ {                                                            │
│   "simu_name": "mes_debug",                                 │
│   "eeg_snr": 5.0,                                           │
│                                                              │
│   "rec_info": {                                              │
│     "fs": 500,           # Sampling frequency (Hz)          │
│     "n_times": 500       # Samples per recording            │
│   },                                                         │
│                                                              │
│   "electrode_space": {                                       │
│     "electrode_montage": "standard_1020",                   │
│     "n_electrodes": 59                                      │
│   },                                                         │
│                                                              │
│   "source_space": {                                          │
│     "src_sampling": "ico3",                                 │
│     "n_sources": 1289,                                      │
│     "constrained_orientation": true                        │
│   }                                                          │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Sample Metadata
```
┌─────────────────────────────────────────────────────────────┐
│ Sample Metadata (JSON per Sample)                            │
├─────────────────────────────────────────────────────────────┤
│ {                                                            │
│   "sample_id": 42,                                           │
│   "active_source_indices": [123, 456, 789],                │
│   "seed_indices": [123, 456, 789],                          │
│   "nb_dipoles": 1,                                          │
│   "scale_ratio": 2.5,                                       │
│   "patches": [[123, 124, 125], [456, 457], [789]]          │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Folder Structure
```
/simulation/fsaverage/
├── constrained/
│   └── standard_1020/
│       └── ico3/
│           ├── simu/
│           │   └── mes_debug/
│           │       ├── mes_debugico3_config.json
│           │       ├── eeg/infdb/
│           │       │   └── *.mat (EEG files)
│           │       └── sources/Jact/
│           │           └── *.mat (source files)
│           │
│           └── model/
│               ├── ch_ico3.mat
│               ├── sources_ico3.mat
│               ├── fwd_ico3-fwd.fif
│               └── checkpoints/
│                   └── best_model.ckpt
│
/results/
├── edf_inference/
│   └── {timestamp}_{filename}/
│       ├── segments/
│       │   └── *.mat
│       ├── animation_data.npz
│       └── summary.json
│
└── mat_inference/
    └── {mat_filename}/
        ├── segments/
        │   └── *.mat
        ├── metrics.json
        ├── animation_data.npz
        └── best_window_summary.json
```

---

## 8. DATA FLOW SUMMARY

### Training Path
```
[Configuration] → [FolderStructure]
       ↓
[HeadModel: ES + SS + FM]
       ↓
[Dataset] ← contains → [Samples]
       ↓
[EEG Signal] + [Source Activity]
       ↓
[NormalizationParams]
       ↓
[TrainingBatch] (batch_size=8)
       ↓
[CNN1Dpl] ← [TrainingConfig]
       ↓
[Checkpoint: best_model.ckpt]
```

### Inference Path
```
[EDF/MAT File] + [NeuralModel Checkpoint]
       ↓
[Segmentation: 500 samples, 50% overlap]
       ↓
[Per-Window Inference]
       ↓
[InferenceResult] × N_windows
       ↓
[MetricsResult] (if ground truth available)
       ↓
[AnimationTimeline] ← [EMA Smoothing α=0.3]
       ↓
[AnimationData.npz] + [Results Summary]
```

---

## 9. QUICK REFERENCE TABLE

| Representation | Type | Shape | Units | File Format |
|---|---|---|---|---|
| **EEG Signal** | ndarray | (59, 500) | μV | EDF/MAT/NPZ |
| **Source Activity** | ndarray | (1289, 500) | A⋅m | MAT |
| **Leadfield** | ndarray | (59, 1289) | V/A | MAT/.fif |
| **ElectrodePosition** | ndarray | (59, 3) | m | MAT |
| **SourcePosition** | ndarray | (1289, 3) | m | MAT |
| **Training Batch** | Tensor | (8, 59, 500) | Norm. | Memory |
| **Prediction** | ndarray | (1289, 500) | A⋅m/Norm. | MAT/NPZ |
| **Animation Timeline** | ndarray | (1289, 40) | Norm. | NPZ |
| **Metrics** | dict/dataclass | - | Mixed | JSON |
| **Config** | dict | - | - | JSON |

---

## 10. KEY CONVERSION FORMULAS

### Normalization
```
max-max:  x_norm = (x - x_min) / (x_max - x_min)
linear:   x_norm = x / |x_max|
global99: x_norm = x / percentile(|x|, 99)
```

### Denormalization
```
x_original = x_norm × max_value  (linear)
x_original = x_norm × (x_max - x_min) + x_min  (max-max)
```

### EMA Smoothing
```
Forward:  y[t] = α × x[t] + (1-α) × y[t-1]
Backward: y[t] = α × y[t] + (1-α) × y[t+1]
```

### Signal Duration
```
duration = n_times / fs
Example: 500 samples @ 500 Hz = 1 second
```

---

## 11. FILE I/O OPERATIONS

### Reading Data
```python
# EEG from MAT
from scipy.io import loadmat
data = loadmat('eeg_file.mat')
eeg = data['EEG']['EEG'][0, 0]  # (59, 500)

# Source from MAT
src = loadmat('source_file.mat')['Jact']  # (1289, 500)

# Animation from NPZ
import numpy as np
anim = np.load('animation_data.npz')
timeline = anim['activity_timeline']  # (1289, 40)
```

### Writing Data
```python
# Save predictions
savemat('predictions.mat', {'predictions': pred_array})

# Save animation
np.savez('animation_data.npz',
    activity_timeline=timeline,
    timestamps=timestamps,
    brain_vertices=vertices,
    brain_faces=faces)
```

---

**Last Updated**: December 17, 2025
**Project**: stESI (Source Reconstruction in EEG using Signal Inference)
