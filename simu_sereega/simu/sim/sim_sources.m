% function Jsrc = sim_sources( lf, src, timeline, viz )
%_________________________________________________________________________%
% src : list of the sources 
% For each sources : 
% index, position, signal

% lf :leadfield

% timeline : epochs object (fs, length, ...)

% viz : either to plot or not the precomputed graphs
%________________________________________________________________________% 

function Jsrc = sim_sources( lf, src, timeline, viz )
% src : list of the sources 
% For each sources : 
% index, position, signal

% lf :leadfield

% timeline : epochs object (fs, length, ...)

% viz : either to plot or not the precomputed graphs
%%% 

    nb_src = numel(src); 
    tmp = size(lf.pos); p = tmp(1);
    
    nb_samples = timeline.srate*timeline.length/1000; 
    
    Jsrc = zeros(p, nb_samples); 
    
    for i = 1:nb_src
        s = src(i); 
        idx = s.idx; 
        
        erp = struct( ...
        'peakLatency', s.sig.center,  ...    % in ms, starting at the start of the epoch
        'peakWidth', s.sig.width,   ...     % in ms
        'peakAmplitude', s.sig.ampl); 

        % check class    
        erp = utl_check_class(erp,'type',s.sig.type);
        erp_ts = generate_signal_fromclass(erp, timeline);
        
        Jsrc(idx,:) = erp_ts; 
    end
        
end