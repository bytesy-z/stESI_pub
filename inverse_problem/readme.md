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