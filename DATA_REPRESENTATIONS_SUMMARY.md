# stESI Data Representations & ER Diagram Summary

## Overview

The stESI project utilizes a comprehensive set of data representations across multiple layers:
- **Anatomical/Biophysical Data**: Head models, electrode positions, source spaces
- **Neurophysiological Data**: EEG signals, source activity (current densities)
- **Machine Learning Data**: Training batches, model parameters, predictions
- **Evaluation Data**: Metrics, segmentation summaries, animation timelines
- **Configuration Data**: Experiment parameters, file structure mappings

---

## Key Data Entity Categories

### 1. **Core Head Model Entities** 📐
| Entity | Purpose | Key Attributes |
|--------|---------|-----------------|
| **HeadModel** | Complete forward model | electrode_space, source_space, leadfield |
| **ElectrodeSpace** | EEG sensor configuration | n_electrodes, positions (n×3), montage_kind |
| **SourceSpace** | Brain source configuration | n_sources, positions (n×3), orientations |
| **ForwardModel** | Forward solution matrix | leadfield (n_elec × n_src), conductivity |

### 2. **Signal Data Entities** 📊
| Entity | Purpose | Dimensions |
|--------|---------|------------|
| **EEGSignal** | Raw recordings | (n_electrodes, n_times) |
| **SourceActivity** | Neural currents | (n_sources, n_times) |
| **NormalizationParams** | Scaling info | max_eeg, max_src, scale_factor |

### 3. **Machine Learning Entities** 🧠
| Entity | Purpose | Role |
|--------|---------|------|
| **Dataset** | Training data container | Holds samples with EEG/source pairs |
| **Sample** | Individual training example | References EEG file, source file, metadata |
| **TrainingBatch** | Mini-batch for training | (batch, n_elec, n_time) tensors |
| **NeuralModel** | 1DCNN/LSTM/DeepSIF | Maps EEG → source predictions |
| **TrainingConfig** | Learning parameters | batch_size, lr, loss, epochs |

### 4. **Inference & Results Entities** 🎯
| Entity | Purpose | Contents |
|--------|---------|----------|
| **InferenceConfig** | Inference parameters | window_samples, overlap, normalization |
| **InferenceResult** | Single window prediction | source_predictions, timestamp |
| **SegmentSummary** | Window metadata | window_index, start_sample, eeg_max |
| **MetricsResult** | Evaluation scores | NMSE, AUC, localization_error, time_error |

### 5. **Visualization Entities** 🎨
| Entity | Purpose | Format |
|--------|---------|--------|
| **AnimationData** | 3D visualization | NPZ with timeline + brain mesh |
| **AnimationTimeline** | Activity over time | (n_sources, n_windows) array |

### 6. **Infrastructure Entities** 🗂️
| Entity | Purpose | Role |
|--------|---------|------|
| **Configuration** | Experiment parameters | simu_name, montage, source_sampling |
| **FolderStructure** | Directory organization | Maps config to physical paths |

---

## Data Flow Relationships

### Training Pipeline
```
Configuration (CFG)
    ↓
FolderStructure (FS) → organizes files
    ↓
Dataset (DS) ← contains
    ↓
Sample (SMP) × N → provides
    ↓
EEGSignal (EEG) + SourceActivity (SRC)
    ↓ [batched]
TrainingBatch (TB) ← uses → NormalizationParams (NP)
    ↓ [forward pass]
NeuralModel (NM) ← trained with → TrainingConfig (TC)
```

### Inference Pipeline
```
NeuralModel (NM) + InferenceConfig (IC)
    ↓
EEGSignal (EEG) → segmented into windows
    ↓
InferenceResult (IR) ← generated (one per window)
    ↓ [if ground truth available]
MetricsResult (MR) ← evaluated
    ↓
AnimationTimeline (AT) → composed of results
    ↓
AnimationData (AD) → packaged (NPZ + mesh)
```

---

## Critical Data Dimensions

### Electrode Space
- **n_electrodes**: 19–64 typical (standard_1020: ~59)
- **Position format**: (n_electrodes, 3) in meters
- **Sampling frequency**: 100–500 Hz typical

### Source Space
- **n_sources**: 1289 (ico3), 5124 (ico4), etc.
- **Position format**: (n_sources, 3) in meters
- **Orientation**: constrained (1D) or unconstrained (3D)

### Signal Dimensions
- **EEG shape**: (n_electrodes, n_times) — typically (59, 500) for 1 sec @ 500 Hz
- **Source shape**: (n_sources, n_times) — typically (1289, 500)
- **Batch shape**: (batch_size, n_electrodes, n_times) — e.g., (8, 59, 500)

### Inference Segmentation
- **Window samples**: 500 typical (1 sec)
- **Overlap fraction**: 0.5 typical (50%)
- **Max windows**: Limited per file (e.g., 100)
- **Output timeline**: (n_sources, n_windows)

---

## Data Normalization Strategies

### Available Methods:
1. **Max-max normalization**: Range [min, max] → [0, 1]
2. **Linear normalization**: Divide by max absolute value
3. **Global 99th percentile**: Use global norm instead of per-window

### Temporal Smoothing:
- **Exponential Moving Average (EMA)**
- **Parameter α**: 0–1, higher = less smoothing
- **⚠️ Warning**: Heavy smoothing (α < 0.3) degrades accuracy ~80%
- **Recommended**: α = 0.5–0.7 for visualization

---

## File Format Summary

### Input Formats:
- **EDF** (.edf): European Data Format for EEG
- **MAT** (.mat): MATLAB binary format
- **NPZ** (.npz): NumPy compressed archive

### Output Formats:
- **NPZ**: Animation data (timeline + mesh)
- **JSON**: Configuration, metadata, metrics
- **MAT**: Segments, predictions

### Storage Locations:
```
/results/
  ├── edf_inference/        # EDF file results
  ├── mat_inference/        # MAT file results
  ├── edf_inference_all/    # Batch results
  └── flash_diagnostic/     # Diagnostic outputs
```

---

## Configuration Structure

```json
{
  "simu_name": "mes_debug",
  "eeg_snr": 5.0,
  "rec_info": {
    "fs": 500,
    "n_times": 500
  },
  "electrode_space": {
    "electrode_montage": "standard_1020",
    "n_electrodes": 59
  },
  "source_space": {
    "src_sampling": "ico3",
    "n_sources": 1289,
    "constrained_orientation": true
  }
}
```

---

## Entity Relationship Diagram (PlantUML)

The `ER_DIAGRAM.puml` file contains a comprehensive ER diagram showing:
- **Entity definitions** with attributes
- **Relationships** between entities (1:1, 1:N, N:M)
- **Data flow** through training and inference pipelines
- **Dependencies** between configuration and execution

### Key Relationships:
- HeadModel **contains** ElectrodeSpace & SourceSpace
- ForwardModel **references** both spaces
- Dataset **contains** Samples
- TrainingBatch **batches** multiple signals
- NeuralModel **trained with** TrainingConfig
- InferenceResult **evaluated with** MetricsResult
- AnimationData **visualizes** AnimationTimeline

---

## Quick Reference: Data Access Patterns

### Loading Head Model
```python
head_model = HeadModel(electrode_space, source_space, folders)
leadfield = head_model.leadfield  # (n_elec, n_src)
```

### Loading Dataset
```python
dataset = EsiDatasetds_new(root_simu, config, ...)
eeg, src = dataset[0]  # (n_elec, n_time), (n_src, n_time)
```

### Creating Batch
```python
dataloader = DataLoader(dataset, batch_size=8)
eeg_batch, src_batch = next(iter(dataloader))
# eeg_batch: (8, n_elec, n_time)
# src_batch: (8, n_src, n_time)
```

### Running Inference
```python
predictions = model(eeg_batch)  # (8, n_src, n_time)
metrics = compute_metrics(predictions, src_batch)
```

---

## Notes

- **Units**: EEG in μV, source activity in A⋅m (or normalized)
- **Precision**: float32/float64 tensors, 64-bit numpy arrays
- **Spatial alignment**: All positions in meters (SI units)
- **Temporal alignment**: All times in seconds; n_times determines duration
