%function [J, M, SNRres, src_info] = sim_simul1(LFmne, LFsereega, src_params,timeline, noise_coeff, SNR, viz)
%________________________________________________________________________
%
%
% - src_params = struct('random_sources', true, 'nb_act_src', nb_act_src,
%     'act_src_idx, act_src_idx, 'act_sig', act_sig_params, 'noise_src_ratio', ratio);
% - act_sig_params = struct('type', 'erp', 'ampl', ampl, 'center', center,
%     'width', width);
%
% noise_coeff = struct('Eyes_artef', 0.05 , 'Anat_noise', 0.1 , 'Colored_noise', 1, 'Elect_noise', 0.1 );
% SNRs = [SNRsrc, SNRmeas];

% src_noise_coef = [0.1, 0.05]; 
%    Jn = src_noise_coef(1)*Jnoise+src_noise_coef(2)*Jarte; 
%    %@TODO : SNR reglable
%    SNRsrc = 10; SNRsrc_db = 10*log10(SNRsrc);
%surf_noise_coef = [1, 0.1]; 
%    N = surf_noise_coef(1)*Ncolored+surf_noise_coef(2)*Nelec; 

    % @TODO NSR reglable :
%    SNRmeas = 10; SNRmeas_db = 10*log10(SNRmeas);
%
%________________________________________________________________________%
function [J, M, SNRres, src_info] = sim_simul1(LFmne, LFsereega, src_params, timeline, noise_coeff, SNR, viz)

    [nb_channels, nb_sources] = size(LFmne.leadfield);
    sPos = LFsereega.pos;
    if src_params.random_sources 
        act_src_idx = lf_get_source_random(LFsereega, src_params.nb_act_src);
    else
        act_src_idx = src_params.act_src_idx;
    end
    act_src_pos = sPos(act_src_idx);
    
    src = []; 
    %@TODO : intoduire variabilité ici
    act_sig = src_params.act_sig; %struct('type', 'erp', 'ampl', 20*10^-9, 'center', 300, 'width', 60);
    for i = 1:src_params.nb_act_src
        new_src = struct('idx', act_src_idx(i), 'pos', sPos(act_src_idx), 'sig', act_sig);
        src = [src; new_src];
    end

    % Simulate source noise : eyeblink and eye movement artefacts...
    % TODO : ajouter nombre d'eye blink réglable/aléatoire
    [Jarte, eb, em, eye_idx, eye_pos] = sim_eyeArtefact( LFmne, LFsereega, timeline, true );
    
    % Bruit brun : 
    nb_noise_sources = nb_sources*src_params.noise_src_ratio; 

    noise_src_idx = lf_get_source_spaced(LFsereega, nb_noise_sources, 1);
    noise_src_pos = sPos(noise_src_idx, :);
    
    noise_src = []; 
    noise_sig = struct('type', 'noise', 'color', 'brown',  'ampl', 1);

    for i = 1:nb_noise_sources
        new_src = struct('idx', noise_src_idx(i), 'pos', sPos(noise_src_idx), 'sig', noise_sig);
        noise_src = [noise_src; new_src];
    end
    
    % Combine sources 
    Jsrc = sim_sources(LFsereega, src, timeline, viz);
    Jnoise = sim_noise_src( LFsereega, noise_src, timeline, true);
    SNRsrc = SNR(1); SNRsrc_db = 10*log10(SNRsrc); 
    
    
    Jn = noise_coeff.anat*Jnoise+noise_coeff.eyes*Jarte; 
    if noise_coeff.anat>0 && noise_coeff.eyes>0
        J = utl_add_snr(Jsrc, Jn, SNRsrc); 
    else
        J = Jsrc;
        SNRsrc = "inf"; SNRsrc_db = "inf";
    end

    %Visualisation
    if viz
        figure()
        subplot(224)
        subplot(221)
        imagesc(Jsrc); 
        title('Active sources')
        subplot(222)
        imagesc(Jarte)
        title('Artefact sources');
        subplot(223)
        imagesc(Jnoise); 
        title('Noise sources'); 
        subplot(224)
        imagesc(J); 
        title('Total sources signal'); 
    end
    % Projection
    Msrc = LFmne.leadfield*J; 

    % Surface noise
    Ncolored = sim_noise_colored(LFsereega, timeline, false);
    Nelec = sim_noise_electronic(LFsereega, timeline, false); 
    
    SNRmeas = SNR(2); SNRmeas_db = 10*log10(SNRmeas);

    N = noise_coeff.colored*Ncolored+noise_coeff.elect*Nelec; 

    M = utl_add_snr(Msrc, N, SNRmeas); 

    [m,i] = max( M(:,300),[], 1);
    if viz
        figure()
        subplot(224)
        subplot(221)
        plot(Msrc(i,:))
        title(strcat('Msrc, EEG measurements without surface noise, channel', " ", num2str(i)))
        subplot(222)
        plot(Ncolored(1,:))
        title('Noise colored')
        subplot(223)
        plot(Nelec(1,:))
        title('Electronic noise')
        subplot(224)
        plot(M(i,:))
        title(strcat(' EEG measurements, channel', " ", num2str(i)))
    end
    
    
    % Output parameters.
    SNRres = struct('SNRsrc', SNRsrc, ...
        'SNRsrc_db', SNRsrc_db', ...
        'SNRmeas', SNRmeas, ...
        'SNRmeas_db', SNRmeas_db);
    src_info = struct('act_src_idx', act_src_idx, 'noise_src_idx', noise_src_idx,...
        'act_src_pos', act_src_pos, 'noise_src_pos', noise_src_pos);
end