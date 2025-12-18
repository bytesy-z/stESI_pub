# stESI Project - Data Representations & Entity Relationships

## Data Representations Used in the Codebase

### 1. **Anatomical/Head Model Data**

#### ElectrodeSpace
- **Type**: Class (`inverse_problem/load_data/HeadModel.py`)
- **Purpose**: Represents electrode configuration and positioning
- **Key Attributes**:
  - `n_electrodes` (int): Number of electrodes
  - `positions` (np.ndarray, shape: n_electrodes × 3): 3D electrode positions
  - `montage_kind` (str): Electrode montage type (e.g., "standard_1020", "easycap-M10")
  - `electrode_names` (List[str]): Channel/electrode labels
  - `electrode_montage` (mne.DigMontage): MNE electrode montage object
  - `info` (mne.Info): MNE info object with electrode metadata
  - `fs` (float): Sampling frequency (Hz)

#### SourceSpace
- **Type**: Class (`inverse_problem/load_data/HeadModel.py`)
- **Purpose**: Represents source space configuration on brain surface
- **Key Attributes**:
  - `src_sampling` (str): Source space sampling (e.g., "ico3", "oct5")
  - `n_sources` (int): Number of sources/dipoles
  - `constrained` (bool): Whether source orientation is constrained (True) or free (False)
  - `positions` (np.ndarray, shape: n_sources × 3): 3D source positions (m)
  - `orientations` (np.ndarray, shape: n_sources × 3): Source orientation unit vectors
  - `surface` (bool): Whether source space is on cortical surface
  - `volume` (bool): Whether source space is volumetric

#### HeadModel
- **Type**: Class (`inverse_problem/load_data/HeadModel.py`)
- **Purpose**: Complete head model combining electrode and source spaces
- **Key Attributes**:
  - `electrode_space` (ElectrodeSpace): Electrode configuration
  - `source_space` (SourceSpace): Source configuration
  - `subject_name` (str): Subject identifier (default: "fsaverage")
  - `fwd` (mne.Forward): MNE forward solution object
  - `leadfield` (np.ndarray, shape: n_electrodes × n_sources): Forward matrix (V/A)

---

### 2. **EEG/Signal Data**

#### EEG Signal Array
- **Type**: np.ndarray or torch.Tensor
- **Format**: `(n_samples, n_electrodes, n_times)` or `(n_electrodes, n_times)`
- **Units**: Microvolts (μV)
- **Data Type**: float32/float64
- **Description**: Raw EEG recordings from electrodes
- **File Formats**: 
  - EDF (European Data Format) - `.edf`
  - MAT (MATLAB) - `.mat`
  - NPZ (NumPy Zipped) - `.npz`

#### Segmented EEG
- **Type**: np.ndarray
- **Format**: `(n_electrodes, window_samples)` for individual windows
- **Purpose**: Divided into overlapping or non-overlapping windows for processing
- **Usage**: Input to neural network models

---

### 3. **Source Data**

#### Source Activity Array
- **Type**: np.ndarray or torch.Tensor
- **Format**: `(n_samples, n_sources, n_times)` or `(n_sources, n_times)`
- **Units**: Ampere-meters (A⋅m) or normalized
- **Data Type**: float32/float64
- **Description**: Neural source amplitudes/current densities
- **Constraint**: Can be constrained (1D) or unconstrained (3D) orientations

#### Active Source Metadata
- **Type**: dict (loaded from JSON)
- **Key Attributes**:
  - `active_indices` (List[int]): Indices of active sources
  - `seed_indices` (List[int]): Indices of seed/primary sources
  - `patches` (List[List[int]]): Grouped source regions
  - `nb_dipoles` (int): Number of dipoles per source

---

### 4. **Dataset & Batch Data**

#### EsiDatasetds_new
- **Type**: torch.utils.data.Dataset class (`inverse_problem/loaders.py`)
- **Purpose**: SEREEGA-based simulation dataset loader
- **Data Per Sample**: 
  - Input: EEG `(n_electrodes, n_times)`
  - Output: Source `(n_sources, n_times)`
- **Key Attributes**:
  - `ids` (List[int]): Sample identifiers
  - `eeg_dict` (dict): Maps id → EEG file path
  - `src_dict` (dict): Maps id → source file path
  - `match_dict` (dict): Maps id → metadata JSON path
  - `md_dict` (dict): Preloaded metadata for each sample
  - `max_eeg` (torch.Tensor): Max values for normalization
  - `max_src` (torch.Tensor): Max values for normalization
  - `snr_db` (float or "random"): Signal-to-noise ratio

#### ModSpikeEEGBuild
- **Type**: torch.utils.data.Dataset class (`inverse_problem/loaders.py`)
- **Purpose**: TVB Neural Mass Model-based dataset
- **Dataset Metadata**:
  - `selected_region` (np.ndarray, shape: n_examples × n_sources × max_size): Spatial patches
  - `nmm_idx` (np.ndarray, shape: n_examples × n_sources): TVB data indices
  - `scale_ratio` (np.ndarray): Magnitude scaling factors
  - `mag_change` (np.ndarray): Magnitude changes within patches
  - `sensor_snr` (np.ndarray, shape: n_examples × 1): Noise levels

---

### 5. **Model Data**

#### CNN1Dpl
- **Type**: torch.nn.Module/pytorch_lightning.LightningModule
- **Architecture**: 1D Convolutional Neural Network
- **Structure**:
  - Conv1D layer: n_electrodes → inter_layer channels
  - Fully connected layer: inter_layer → n_sources
- **Parameters**:
  - `channels` (List[int]): [n_electrodes, inter_layer, n_sources]
  - `kernel_size` (int): Convolution kernel size (e.g., 5)
  - `bias` (bool): Whether to use bias terms
- **Input Shape**: `(batch_size, n_electrodes, n_times)`
- **Output Shape**: `(batch_size, n_sources, n_times)`

#### Training Batch
- **Type**: Tuple[torch.Tensor, torch.Tensor]
- **Format**: `(eeg_batch, source_batch)`
  - `eeg_batch` (torch.Tensor, shape: batch_size × n_electrodes × n_times)
  - `source_batch` (torch.Tensor, shape: batch_size × n_sources × n_times)

---

### 6. **Inference Results**

#### MetricsResult
- **Type**: @dataclass (`inverse_problem/run_mat_inference.py`)
- **Purpose**: Evaluation metrics for inference output
- **Attributes**:
  - `nmse` (float): Normalized Mean Squared Error
  - `auc` (float): Area Under Curve (ROC)
  - `localization_error_mm` (float): Localization error in mm
  - `time_error_ms` (float): Temporal error in ms
  - `seed_indices` (List[int]): Ground truth active source indices
  - `estimated_seed_indices` (List[int]): Predicted active source indices
  - `peak_times_gt` (List[int]): Ground truth peak times (samples)
  - `peak_times_pred` (List[int]): Predicted peak times (samples)

#### SegmentSummary
- **Type**: @dataclass (`inverse_problem/run_mat_inference.py`)
- **Purpose**: Metadata for processed EEG segments
- **Attributes**:
  - `index` (int): Window/segment index
  - `start_sample` (int): Starting sample in original signal
  - `start_time_seconds` (float): Starting time
  - `eeg_max_abs` (float): Maximum absolute amplitude
  - `mat_path` (Path): Output file path

#### Window Prediction
- **Type**: dict
- **Format**:
  ```python
  {
    "window_index": int,
    "predictions": np.ndarray,  # shape: (n_sources, n_times)
    "timestamp": float,
    "eeg_segment": np.ndarray,  # shape: (n_electrodes, n_times)
    "metrics": MetricsResult (if ground truth available)
  }
  ```

#### Animation Timeline
- **Type**: Tuple[np.ndarray, np.ndarray]
- **Format**:
  - Activity timeline: `(n_sources, n_windows)` - source activity per window
  - Timestamps: `(n_windows,)` - time for each window (seconds)

---

### 7. **Configuration Data**

#### General Config Dictionary
- **Type**: dict (loaded from JSON)
- **File**: `{simu_name}{src_sampling}_config.json`
- **Key Sections**:
  ```
  {
    "simu_name": str,
    "eeg_snr": float or "infdb",
    "rec_info": {
      "fs": float,  # sampling frequency
      "n_times": int  # samples per recording
    },
    "electrode_space": {
      "electrode_montage": str,
      "n_electrodes": int
    },
    "source_space": {
      "src_sampling": str,
      "n_sources": int,
      "constrained_orientation": bool
    }
  }
  ```

#### Folder Structure
- **Type**: FolderStructure class (`inverse_problem/load_data/FolderStructure.py`)
- **Purpose**: Organizes data directory hierarchy
- **Key Paths**:
  - `data_folder`: Root simulation data
  - `eeg_folder`: EEG recordings directory
  - `source_folder`: Source activity directory
  - `model_folder`: Trained model checkpoints

---

### 8. **Normalized/Processed Data**

#### Smoothed Activity Timeline
- **Type**: np.ndarray
- **Format**: `(n_sources, n_windows)`
- **Method**: Exponential Moving Average (EMA) with bidirectional smoothing
- **Parameters**:
  - `alpha` (float, 0-1): Smoothing factor (higher = less smoothing)
  - Warning: Heavy smoothing (α < 0.3) degrades accuracy ~80%

#### Normalized Signals
- **Type**: Tensor/Array
- **Methods**:
  - **Max-max normalization**: Range [min, max] → [0, 1]
  - **Linear normalization**: Scale by global maximum
  - **Global 99th percentile**: Uses global norm instead of per-window

---

### 9. **Frontend Data (TypeScript)**

#### API Request
- **Type**: FormData with File
- **Endpoint**: `/api/analyze-eeg` or `/api/analyze-mat`
- **Contents**:
  - `file`: EDF or MAT file (binary)

#### API Response
- **Type**: JSON
- **Format**:
  ```json
  {
    "success": boolean,
    "message": string,
    "results_path": string,
    "animation_data": {
      "activity_timeline": number[][],
      "timestamps": number[],
      "vertices": number[][],
      "faces": number[][]
    }
  }
  ```

#### Animation Data (NPZ)
- **Type**: NumPy compressed archive
- **Contents**:
  - `activity_timeline`: Source activity over time
  - `timestamps`: Time points
  - `vertices`: Brain mesh vertices
  - `faces`: Brain mesh face indices

---

## Summary Table

| Data Type | Format | Dimensions | Units | Purpose |
|-----------|--------|-----------|-------|---------|
| EEG Signal | np.ndarray | (n_electrodes, n_times) | μV | Raw brain recordings |
| Source Activity | np.ndarray | (n_sources, n_times) | A⋅m | Neural current estimates |
| Leadfield | np.ndarray | (n_electrodes, n_sources) | V/A | Forward model matrix |
| Batch (Training) | torch.Tensor | (batch, electrodes, time) | Mixed | Model input/output |
| Predictions | np.ndarray | (n_sources, n_times) | A⋅m or norm. | Network output |
| Metrics | MetricsResult | - | Mixed | Evaluation scores |
| Timeline (Animation) | np.ndarray | (n_sources, n_windows) | Normalized | 3D visualization data |
| Config | dict | - | - | Experiment parameters |
