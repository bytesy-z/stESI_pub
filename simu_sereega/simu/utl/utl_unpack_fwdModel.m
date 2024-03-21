% Function to get the data saved from mne-python forward model generation



function [CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, source_constrained )

    % loading information from mne-python for simulation
    CH     = load(strcat(folder_path, '/ch_', suf, '.mat'));
    CHLOCS = load(strcat(folder_path, '/chlocs_', suf, '.mat'));
    SRC    = load(strcat(folder_path, '/sources_', suf, '.mat'));
    LF     = load(strcat(folder_path, '/LF_', suf, '.mat'));

    % some important parameters
    nb_sources  = SRC.nb_sources; 
    nb_channels = CH.nb_channels;
    
    % Reshape chanlocs argument (for leadfield definition)
    chanlocs = utl_unpack_chanlocs(nb_channels, CHLOCS); 
    

    
    % LEADFIELD?
    % Reshape the leadfield matrix so that the dimension match between
    % mne-python and sereega for both constrained sources and unconstrained
    % sources
    LFmne = struct( ...
        'leadfield', LF.G, ...
        'orientation', SRC.orientations, ...
        'pos', SRC.positions, ...
        'chanlocs', chanlocs);
    
    source_orientations = SRC.orientations; 
    source_positions    = SRC.positions; 

    [G, oris, spos] = utl_reshape_leadfield(source_constrained, LF.G, ...
        source_positions, source_orientations, nb_channels, nb_sources); 

    source_positions = spos; source_orientations = oris; 
    clear spos; 
    clear oris; 

    source_positions = source_positions * 10^3;
    lf = struct( ...
        'leadfield', G, ...
        'orientation', source_orientations, ...
        'pos', source_positions, ...
        'chanlocs', chanlocs);
    
    SRC.positions = source_positions; 
    
    LFsereega = lf; clear lf; 
    
end
