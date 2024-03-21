% function Jn = sim_noise_src( lf, src, timeline, viz )
%________________________________________________________________________%
% src : list of the sources 
% For each sources : 
% index, position, signal

% lf :leadfield

% timeline : epochs object (fs, length, ...)

% viz : either to plot or not the precomputed graphs
%_________________________________________________________________________%

function Jn = sim_noise_src( lf, src, timeline, viz )


    nb_src = numel(src); 
    tmp = size(lf.pos); p = tmp(1);
    
    nb_samples = timeline.srate*timeline.length/1000; 
    
    Jn = zeros(p, nb_samples); 
    
    for i = 1:nb_src
        s = src(i); 
        idx = s.idx; 
        
        noise = struct( ...
        'type', 'noise', ...
        'color', s.sig.color, ...
        'amplitude', s.sig.ampl, ...
        'peakAmplitudeDv', 20);
        noise = utl_check_class(noise);
        

        noise_ts = generate_signal_fromclass(noise, timeline);
        
        Jn(idx,:) = noise_ts; 
    end
        
end