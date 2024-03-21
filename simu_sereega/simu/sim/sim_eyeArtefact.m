function [Jarte, eb, em, eye_idx, eye_pos] = sim_eyeArtefact( LFmne, LFsereega, timeline, viz )

    [eye_idx, eye_pos] = get_eyes_src(LFsereega);
    
    nb_samples = timeline.length*timeline.srate/1000;
    [nb_channels, nb_dipoles] = size(LFmne.leadfield); %
    Jarte = zeros(nb_dipoles, nb_samples); 
    
    tvec = linspace(0, timeline.length/1000-1/timeline.srate, ...
        timeline.length*timeline.srate/1000);
    
    ti = 0; tf = timeline.length/1000; 
    eb = sim_eyeBlink(timeline.srate, tvec); 
    em = sim_eyeMvt(timeline.srate, tvec, ti, tf, viz); 
    
    if viz
        figure()
        subplot(212)
        subplot(211)
        plot(tvec, eb); 
        title('Eye blink time course'); 
        subplot(212)
        plot(tvec, em); 
        title('Eye movement time course'); 
        sgtitle('Eyes artefacts time courses'); 
    end
    
    Jarte(eye_idx, :) = repmat( 1/2*(eb+em), numel(eye_idx), 1); 
    
end