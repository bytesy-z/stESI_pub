%function c = get_component_Grech(lf, src_idx, timeline, viz)
% _______________________________________________________________________
% Inputs : 
% - lf : leadfield matrix (sereega)
% - src_idx : index of the source to use to create the component
% - dev  : dev(1) = amplitude deviation coefficient, dev(2) = width
% deviation coefficient, dev(3) = latency deviation coefficient
% - viz : [true or false], Optional (default is false) -> visualize the 
%    source position and the signal time serie? 
% 
% Output : 
% - c : the component created with an erp signal of parameters : 
%       'peakLatency', 500,  ...    % in ms, starting at the start of the
%        epoch
%       'peakWidth', 20,   ...     % in ms
%       'peakAmplitude', 1, ... % 1V.
%       'peakAmplitudeDv', 50);  % 50% amplitude deviation possible btwn
%       trials.
% 
% ________________________________________________________________________%
function c = get_component_Grech(lf, src_idx, timeline, erp_params, erp_dev, viz)

    if nargin<5
        viz = false;
    end

    % Signal definition
    ampl   = erp_params.ampl; 
    center = erp_params.center; 
    width  = erp_params.width; 
    
     
    %type = 'erp';
    if ~ isempty(erp_dev)
        ampl_dev   = erp_dev.ampl; 
        width_dev  = erp_dev.width; 
        center_dev = erp_dev.center;
        erp = struct( ...
            'peakLatency', center,  ...    % in ms, starting at the start of the epoch
            'peakWidth', width,   ...     % in ms
            'peakAmplitude', ampl, ...
            'peakAmplitudeDv', ampl_dev*ampl, ...
            'peakWidthDv',     width_dev*width, ...
            'peakLatencyDv',   center_dev*center); 
        %disp('Dev parameter'); 
    else 
        erp = struct( ...
            'peakLatency', center,  ...    % in ms, starting at the start of the epoch
            'peakWidth', width,   ...     % in ms
            'peakAmplitude', ampl, ...
            'peakAmplitudeDv', 0.5*ampl); 
    end

    erp = utl_check_class(erp,'type','erp');
    if viz
        plot_source_location(src_idx, lf);
        title('Source visualization')
    
        plot_signal_fromclass(erp, timeline);
        xlabel('Time [ms]'); ylabel('Amplitude [µV]');
        title('Simulated ERP'); 
    end

    % Component creation
    c        = struct();
    c.source = src_idx;       
    c.signal = {erp};
    c        = utl_check_component(c, lf);

end