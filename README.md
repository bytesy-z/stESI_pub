# Environment
```
conda env create -n pt_env python=3.9
conda activate pt_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mne
pip install pytorch-lightning
pip install colorednoise
pip install tensorboard
```

# Create head model
`simu_head_model/`

Head Model gathers
- source positions and orientations
- electrode positions
- BEM model (conductor model)
- Leadfield matrix
- triangle and vertices of the brain mesh
It is created using the mne-python library.

`create_head_model.py`
Example of command line to create a head model with the following parameters:  
- subject: fsaverage
- source space subsampling: ico3
- electrode montage: standard_1020
- sources with constrained (fixed) orientation
- default conductivity values = (0.3, 0.006, 0.3)
- save to save the model
```
python create_head_model.py -subject_name fsaverage -source_sampling ico3 -electrode_montage standard_1020 -constrained -conductivity 0.3 0.006 0.3 -save
```

*Nota* : 
- 2 subjects are currently supported: fsaverage and sample (*to come: deepsif*)
- 2 electrode montage are currently supported: standard_1020 and sample (however it is not difficult to add a new one if it is available in mne-python)
- *to come: pour le cas deepsif : retourne deux matrices de leadfield -> region qui correspond au leadfield obtenu en plaçant un dipole par région et summed qui correspond au leadfield obtenu en sommant les dipoles de la région (comme c'est fait dans deepsif)

# SEREEGA simulations
`simu_sereega/`

**Matlab**
**First**: download SEREEGA toolbox at https://github.com/lrkrol/SEREEGA 
Put the SEREEGA codes in the same folder as SEREEGA simulations code (or anywhere else but change `sereega_add_path.m` script).

Single or multiple extended sources simulations (same script), with "event related" like activity.

/!/ run sereega_add_path.m before trying to launch other scripts.

A lot of parameters in this script which can be changed, about the head model, the spatial and temporal pattern.
Important parameters of the simulation are : 
*head model to use* : had to be simulated first (c.f above).
*spatial pattern*
- order_min, order_max: min and max extension order of a region of neighboring active sources
- n_patch_min, n_patch_max: min and max number of active ragions of neighboring sources
*temporal patter*
- amplitude, width and center: of the Gaussian signal used to simulate the waveform. 
- dev parameters for amplitude, width and center: controls the variability of the dataset

## if you want to manipulate data but not use matlab
`simu_source_python/`

-> python code to simulate data which should be similar to the sereega simulations. 
`simu_extended_source.py`
- Give the information about the (previously simulated) head model to use (subject name, elecrode montage, source subsampling, orientation)
- Give a name for the simulation
- Timeline parameters (sampling frequency, duration in ms)
- Spatial and temporal parameters for the source simulation 
Example :
```
python simu_extended_source.py -sin mes_debug_python -ne 100 -mk standard_1020 -ss ico3 -o constrained -sn fsaverage -fs 512 -d 500 -m 2 -np_min 1 -np_max 3 -o_min 1 -o_max 5 -amp 10 -w 60
```


# NMM based data simulation 
original code from: https://github.com/bfinl/DeepSIF



# References 
- mne-python: https://mne.tools/stable/index.html
- SEREEGA: Krol, L. R., Pawlitzki, J., Lotte, F., Gramann, K., & Zander, T. O. (2018). SEREEGA: Simulating Event-Related EEG Activity. Journal of Neuroscience Methods, 309, 13-24.
- LSTM network and simulation of extended sources : 
    - https://github.com/LukeTheHecker/esinet
    - Hecker L, Rupprecht R, Tebartz Van Elst L and Kornmeier J (2021) ConvDip: A Convolutional Neural Network for Better EEG Source Imaging. Front. Neurosci. 15:569918. doi: 10.3389/fnins.2021.569918
    - Hecker L., Rupprecht R., Tebartz van Elst L., Kornmeier J., Long-Short Term Memory Networks for Electric Source Imaging with Distributed Dipole Models, bioRxiv 2022.04.13.488148; doi: https://doi.org/10.1101/2022.04.13.488148
- deepSIF network and NMM simulations: 
    - Sun R, Sohrabpour A, Worrell GA, He B: “Deep Neural Networks Constrained by Neural Mass Models Improve Electrophysiological Source Imaging of Spatio-temporal Brain Dynamics.” Proceedings of the National Academy of Sciences of the United States of America 119.31 (2022): e2201128119.
    - https://github.com/bfinl/DeepSIF 
    - seems to be an updated version of the codes: https://github.com/RuifengZheng/DeepSIF
    - NMM data simulations: 
        - source code: https://github.com/the-virtual-brain/tvb-root
        - doc: https://docs.thevirtualbrain.org/
