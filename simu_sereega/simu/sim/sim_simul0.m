% function [J, M, resSNR, src_info] = sim_simul0(LFmne, LFsereega, ...
%             src, timeline, act_src_params, act_sig_params, viz)
%_________________________________________________________________________%
% Noise free simulation (to check amplitudes of signal)
% Simulation of 1 time series (for "raw" data type in mne).
%
% The signal is obtained using the projection of source data via the
% leadfield matrix. 
% M = GJ + n

% Input 
% - LFmne : leafidle structure, containing the leadfield from mne 
% - LFsereega : leadfield structure containing the reshape leadfield suited
% for the functions in sereega
% - src : 
% - timeline : epochs structure containing the timeline parameters of the
% simulation (sampling frequency, length (in ms), number of epochs = 1)
% - act_src_params : nb, idx, pos of active sources
%    -> act_src_params = struct('nb_src', nb_act_src, 'idx', act_src_idx, 'pos', act_src_pos);
% - act_sig_params : parameters of activation signal
%    -> act_sig_params = struct('type', type, 'ampl', ampl, 'center', center, 'width' width);
% - viz : true or false, to plot or not default graph during computation
%
% Output 
% J : source vector (dim nb_dipoles, nb_samples)
% M : simulated EEG measurements (nb_channels, nb_samples)
% resSNR : vector of SNRs (all = "inf" since no noise)
% src_info : structure containing information about the sources used in J
% (indexes, signal time course, position)
% _______________________________________________________________________%

function [J, M, resSNR, src_info] = sim_simul0(LFmne, LFsereega, src, timeline, act_src_params, act_sig_params, viz)
    
    act_src_idx = act_src_params.idx; 
    act_src_pos = act_src_params.pos;
    
        
    % ERP
    erp = struct( ...
        'peakLatency', act_sig_params.center,  ...    % in ms, starting at the start of the epoch
        'peakWidth', act_sig_params.width,   ...     % in ms
        'peakAmplitude', act_sig_params.ampl); 

    % check class    
    erp = utl_check_class(erp,'type',act_sig_params.type);
    if viz
        plot_source_location(act_src_idx, LFsereega);
        title('Source visualization')
        
        plot_signal_fromclass(erp, timeline);
        xlabel('Time [ms]'); ylabel('Amplitude [µV]');
        title('Simulated ERP'); 
    end
    
    % Signal generation
    erp_ts    = generate_signal_fromclass(erp, timeline); 
    
    % J 
    if src.constrained
        nb_dipoles = src.nb_sources; 
    else
        nb_dipoles = 3*src.nb_sources; 
    end
    nb_samples = timeline.length*timeline.srate/1000; 
    J = zeros(nb_dipoles, nb_samples, timeline.n); 
    
    if act_src_params.nb_src > 1
        J(act_src_idx, :,:) = repmat(erp_ts, act_src_params.nb_src, 1);
    else
        J(act_src_idx, :, :)      = erp_ts;
    end
    
    % Projection i.e computation of eeg measurements.
    G = LFmne.leadfield;
    M = G*J;
    
    
    % SNR (juste to match other simulations)
    SNRmeas = "inf"; 
    SNRmeas_db = "inf";
    SNRsrc = "inf"; SNRsrc_db = "inf";

    % Output parameters.
    noise_src_idx = []; noise_src_pos = []; 
    resSNR = struct('SNRsrc', SNRsrc, ...
        'SNRsrc_db', SNRsrc_db', ...
        'SNRmeas', SNRmeas, ...
        'SNRmeas_db', SNRmeas_db);
    src_info = struct('act_src_idx', act_src_idx, 'noise_src_idx', noise_src_idx,...
        'act_src_pos', act_src_pos, 'noise_src_pos', noise_src_pos);
    
    
end
