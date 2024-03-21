## build simple fsaverage model 
#python create_head_model.py -subject_name fsaverage -source_sampling ico3 -constrained -conductivity 0.3 0.006 0.3 -electrode_montage standard_1020 -save

## build sample head model (for real data experiments)
#python create_head_model.py -subject_name sample -source_sampling ico3 -constrained -conductivity 0.3 0.006 0.3 -save

## build deepsif based head model (NMMs experiments)
python create_head_model.py -subject_name deepsif -source_sampling fsav_994 -deepsif_data Documents/deepsif/DeepSIF-Main/ -constrained -conductivity 0.33 0.004125 0.33 -electrode_montage standard_1020 -save
