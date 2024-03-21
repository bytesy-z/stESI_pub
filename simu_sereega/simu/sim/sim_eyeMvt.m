function [eye_mvt_data, tvec] = sim_eyeMvt( fs, tvec, ti, tf, viz)

    % Duration of the eye movement : 
    mvt_duration = (tf-ti)*rand(1)+ti; 
    % Corresponding number of samples : 
    Tm = ceil(fs*mvt_duration);

    % Start of the movement (i.e of the offset) :
    % start btwen 0 and fs*tf-Tm
    mvt_start = ceil(  (fs*tf-Tm)*  rand(1) );
    

    nb_samples = length(tvec);
    eye_mvt_data = zeros(size(tvec));
    eye_mvt_data( mvt_start:mvt_start+Tm-1 ) = ones(Tm, 1);

%     if viz
%         figure(); 
%         plot(tvec, eye_mvt_data);
%         title('Eye mvt artefact')
%     end

end