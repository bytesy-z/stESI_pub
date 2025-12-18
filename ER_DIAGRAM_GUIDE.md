# stESI Entity Relationship Diagram - PlantUML Code

Use this PlantUML diagram to visualize the data entities and their relationships in the stESI project.

## How to Use

1. **Online**: Copy the PlantUML code to [PlantUML Editor](https://www.plantuml.com/plantuml/uml/)
2. **VS Code**: Install the [PlantUML extension](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml) and preview with `Alt+D`
3. **Command Line**: 
   ```bash
   plantuml ER_DIAGRAM.puml -o output.png
   ```

## PlantUML Code

```plantuml
@startuml stESI_Entity_Relationship_Diagram
!theme plain
skinparam backgroundColor #f5f5f5
skinparam linetype ortho

' ============================================================================
' CORE HEAD MODEL ENTITIES
' ============================================================================

entity "HeadModel" as HM {
  *subject_name: str
  --
  electrode_space: ElectrodeSpace
  source_space: SourceSpace
  fwd: mne.Forward
  leadfield: ndarray (n_electrodes × n_sources)
}

entity "ElectrodeSpace" as ES {
  *montage_id: str
  --
  n_electrodes: int
  positions: ndarray (n_electrodes × 3)
  montage_kind: str
  electrode_names: List[str]
  electrode_montage: mne.DigMontage
  fs: float
  info: mne.Info
}

entity "SourceSpace" as SS {
  *src_sampling: str
  --
  n_sources: int
  constrained: bool
  positions: ndarray (n_sources × 3)
  orientations: ndarray (n_sources × 3)
  surface: bool
  volume: bool
}

entity "ForwardModel" as FM {
  *model_id: str
  --
  leadfield: ndarray (n_electrodes × n_sources)
  electrode_space_ref: str
  source_space_ref: str
  method: str (BEM/FEM)
  conductivity: Tuple[float, float, float]
}

' ============================================================================
' SIGNAL DATA ENTITIES
' ============================================================================

entity "EEGSignal" as EEG {
  *signal_id: str
  --
  data: ndarray (n_electrodes × n_times)
  fs: float
  duration_sec: float
  units: str (μV)
  file_format: str (EDF/MAT/NPZ)
  snr_db: float
}

entity "SourceActivity" as SRC {
  *source_id: str
  --
  data: ndarray (n_sources × n_times)
  units: str (A⋅m or norm)
  orientation: str (constrained/unconstrained)
  active_indices: List[int]
  seed_indices: List[int]
}

entity "NormalizationParams" as NP {
  *norm_id: str
  --
  method: str (max-max/linear/global99)
  max_eeg: float
  max_src: float
  scale_factor: float
}

' ============================================================================
' MACHINE LEARNING ENTITIES
' ============================================================================

entity "TrainingBatch" as TB {
  *batch_id: str
  --
  eeg_signals: Tensor (batch × n_elec × n_time)
  source_activities: Tensor (batch × n_src × n_time)
  batch_size: int
  max_value_eeg: Tensor
  max_value_src: Tensor
  normalization: str
}

entity "NeuralModel" as NM {
  *model_id: str
  --
  architecture: str (1DCNN/LSTM/DeepSIF)
  n_electrodes: int
  n_sources: int
  kernel_size: int
  intermediate_channels: int
  parameters: int
  checkpoint_path: str
}

entity "TrainingConfig" as TC {
  *training_id: str
  --
  model_architecture: str
  batch_size: int
  learning_rate: float
  n_epochs: int
  loss_function: str (cosine/MSE/logMSE)
  optimizer: str (Adam)
  early_stopping: bool
}

' ============================================================================
' DATASET ENTITIES
' ============================================================================

entity "Dataset" as DS {
  *dataset_id: str
  --
  dataset_type: str (SEREEGA/NMM)
  n_samples: int
  n_electrodes: int
  n_sources: int
  n_times: int
  snr_db: float
  noise_type: str
  normalization: str
}

entity "Sample" as SMP {
  *sample_id: int
  --
  eeg_file: str
  source_file: str
  metadata_file: str
  active_source_indices: List[int]
  scale_ratio: float
}

' ============================================================================
' INFERENCE & RESULTS ENTITIES
' ============================================================================

entity "InferenceConfig" as IC {
  *inference_id: str
  --
  model_checkpoint: str
  window_samples: int
  overlap_fraction: float
  use_global_norm: bool
  smoothing_alpha: float
  max_windows: int
}

entity "InferenceResult" as IR {
  *result_id: str
  --
  window_index: int
  source_predictions: ndarray (n_sources × n_times)
  confidence: float
  timestamp: float
  processing_time_ms: float
}

entity "SegmentSummary" as SEG {
  *segment_id: str
  --
  window_index: int
  start_sample: int
  start_time_sec: float
  eeg_max_abs: float
  mat_path: str
}

entity "MetricsResult" as MR {
  *metrics_id: str
  --
  nmse: float
  auc: float
  localization_error_mm: float
  time_error_ms: float
  seed_indices: List[int]
  estimated_seed_indices: List[int]
  peak_times_gt: List[int]
  peak_times_pred: List[int]
}

' ============================================================================
' VISUALIZATION ENTITIES
' ============================================================================

entity "AnimationTimeline" as AT {
  *timeline_id: str
  --
  activity: ndarray (n_sources × n_windows)
  time_points: ndarray (n_windows)
  smoothing_alpha: float
}

entity "AnimationData" as AD {
  *animation_id: str
  --
  activity_timeline: ndarray (n_sources × n_windows)
  timestamps: ndarray (n_windows)
  brain_vertices: ndarray (n_vertices × 3)
  brain_faces: ndarray (n_faces × 3)
  format: str (NPZ)
}

' ============================================================================
' INFRASTRUCTURE ENTITIES
' ============================================================================

entity "Configuration" as CFG {
  *config_id: str
  --
  simu_name: str
  electrode_montage: str
  source_sampling: str
  constrained: bool
  fs: float
  n_times: int
  eeg_snr: float
  orientation: str
}

entity "FolderStructure" as FS {
  *folder_root: str
  --
  root_folder: str
  data_folder: str
  model_folder: str
  eeg_folder: str
  source_folder: str
  active_source_folder: str
}

' ============================================================================
' RELATIONSHIPS
' ============================================================================

' HeadModel relationships
HM ||--|| ES : "contains"
HM ||--|| SS : "contains"
HM ||--|| FM : "uses"

' ForwardModel relationships
FM ||--|| ES : "references"
FM ||--|| SS : "references"

' Signal relationships
EEG ||--|| HM : "measured with"
SRC ||--|| HM : "defined in"

' Normalization relationships
SRC -->o NP : "normalized by"
EEG -->o NP : "scaled by"
TB -->o NP : "uses"
IR -->o NP : "applies"

' Training relationships
TB ||--}o EEG : "batches from"
TB ||--}o SRC : "batches from"
TB -->o NM : "trains"
NM }o--|| TC : "trained with"

' Dataset relationships
DS }o--|| CFG : "configured by"
DS ||--|{ SMP : "contains"
SMP ||--|| EEG : "provides"
SMP ||--|| SRC : "provides"

' Inference relationships
IR ||--|| NM : "produced by"
IR ||--|| EEG : "infers from"
IR }o--|| IC : "runs with"
SEG ||--|| IR : "summarizes"
SEG ||--|| EEG : "segments"

' Metrics relationships
MR ||--|| IR : "evaluates"
MR ||--|| SRC : "compares to"
SMP ||--|| MR : "evaluated with"

' Visualization relationships
AD ||--|| IR : "visualizes"
AD ||--|| AT : "contains"

' Infrastructure relationships
FS }o--|| DS : "organizes"
FS ||--|| CFG : "structures"

@enduml
```

## Diagram Components

### Legend
- **|**: One entity
- **}**: Many entities
- **--**: 1-to-1 relationship
- **--|{**: 1-to-many relationship
- **}o--**: Many-to-1 relationship

### Color Coding by Layer
- **Head Model** (Core anatomical data)
- **Signal Data** (EEG and source activity)
- **Machine Learning** (Models and training)
- **Dataset Management** (Data organization)
- **Inference & Results** (Processing outputs)
- **Visualization** (3D animation data)
- **Infrastructure** (Configuration and file management)

## Key Data Flows

### 1. Forward Model Generation
```
ElectrodeSpace + SourceSpace → ForwardModel → HeadModel
```

### 2. Training Pipeline
```
Configuration → Dataset → Samples → TrainingBatch → NeuralModel
                                                      ↓
                                                  TrainingConfig
```

### 3. Inference Pipeline
```
NeuralModel + InferenceConfig + EEGSignal → InferenceResult → MetricsResult
                                                ↓
                                           SegmentSummary
                                                ↓
                                          AnimationTimeline
                                                ↓
                                            AnimationData
```

### 4. Data Normalization Flow
```
EEG/SRC + NormalizationParams → TrainingBatch/InferenceResult
```

## Usage Examples

### Loading and Preparing Data
```
FolderStructure → Configuration
    ↓
HeadModel (ES + SS + FM)
    ↓
Dataset → Samples → EEG/SRC
    ↓
NormalizationParams
    ↓
TrainingBatch
```

### Running Inference
```
NeuralModel ← TrainingConfig (checkpoint)
    ↓
InferenceConfig + EEGSignal
    ↓
InferenceResult (per window)
    ↓
SegmentSummary + MetricsResult (if GT available)
    ↓
AnimationTimeline → AnimationData (visualization)
```

## Related Files
- `DATA_REPRESENTATIONS.md`: Detailed entity descriptions
- `DATA_REPRESENTATIONS_SUMMARY.md`: Summary with tables and examples
- `ER_DIAGRAM.puml`: Original PlantUML file
