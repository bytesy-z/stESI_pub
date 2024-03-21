function N = sim_noise_electronic(lf, timeline, viz)
    
    [nb_channels, ~] = size(lf.leadfield); 
    nb_samples = timeline.length*timeline.srate/1000; 
    N = randn(nb_channels, nb_samples); 
end