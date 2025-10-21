## Environment
Create a virtual environment with pytorch, pytorch-lightning, mne-python (basic functionnalities) and colorednoise.
```
conda env create -n pt_env
conda activate pt_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mne
pip install pytorch-lightning
pip install colorednoise
pip install tensorboard
```

## Training
Three models available: 1dcnn, lstm and deepsif.
- In the main_train.py : change the path to the data repository (path for SEREEGA data, NMM data and results)
- Then run the code with the proper parameters 
Example for the 1dcnn, on the _dps_sereega_2src_om3_train_big_ dataset, with 100 samples : 

```
python main_train.py _dps_sereega_2src_om3_train_big_ sereega -source_space fsav_994 -spikes_folder nmm_spikes_nov23_train -model 1dcnn -to_load 100 -eeg_snr 5 -inter_layer 2048 -kernel_size 5 -batch_size 8 -n_epochs 100 -scaler linear -loss cosine -sfolder train_size_impact
```
## Eval
Example of command for evaluation : 
```
python eval.py _dps_sereega_2src_om3_test_ sereega -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -spikes_folder nmm_spikes_nov23_test -eeg_snr 5 -to_load 2000 -per_valid 1 -sfolder train_size_impact_res -train_simu_name _dps_sereega_2src_om3_train_big_ -train_simu_type sereega -n_train_samples 3200 -train_sfolder train_size_impact
```

## EDF → 1dCNN inference pipeline

`run_edf_inference.py` converts a long-form EDF recording into the `.mat` layout consumed by the 1dCNN loader, evaluates the network on every window, and exports both the converted segments and an interactive cortical heatmap highlighting the highest-activation window.

```
python run_edf_inference.py /path/to/recording.edf \
	--simu_name mes_debug \
	--subject fsaverage \
	--electrode_montage standard_1020 \
	--source_space ico3 \
	--inter_layer 4096 \
	--kernel_size 5 \
	--open_plot
```

- The script resamples data to the training rate (default 512 Hz), converts the linked-ear reference to an average reference, and interpolates any missing 10-20 channels so the tensor matches the 90-channel montage used during training.
- EEG is segmented into windows that match the training temporal length (256 samples, i.e. 0.5 s); each window is saved as `segments/segment_XXXX.mat` using the same `eeg_data.EEG` structure as the simulated datasets. `segments_metadata.json` summarises the start time, scaling factor, and filename for every window.
- The window with the highest predicted activity energy is stored in `best_window_summary.json`, and its interactive Plotly brain map is written next to the segments. Add `--open_plot` if you want the browser to launch automatically.
- Use `--window_seconds`, `--overlap_fraction`, and `--max_windows` to customise segmentation density. `--output_dir` lets you choose a different destination inside the repository tree.

> **Tip:** Place the EDF file somewhere inside the repository (for example under `sample/`) so the generated artefacts remain under version control.