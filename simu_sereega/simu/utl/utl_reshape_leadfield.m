function [Gsereega, oris, spos] = utl_reshape_leadfield(source_constrained, Gmne,source_positions, source_orientations, nb_channels, nb_sources)
    % In both cases (source_constrained or unconstrained)

    if source_constrained
        Gsereega = zeros(nb_channels, nb_sources, 3); 
        % Then 1 source = 1 dipole oriented orthogonally to the surface
        % Problem : the sereega leadfield is always decomposed in 3
        % components (for the 3 directions) and the mne leadfield in the
        % case of oriented sources is already a sum of the value in the
        % three directions since they are known and fixed
        Gsereega(:,:,:) = repmat(Gmne,1,1,3); 
        oris = (1/3.0)*ones(nb_sources,3); 
        spos = source_positions; 
    else
        Gsereega = zeros(nb_channels, 3*nb_sources, 3); 
        % unconstrained sources : 1 sources = 3 dipoles forming a
        % orthogonal basis. The mne leadfiled is of dimension nb_channles,
        % 3*nb_sources (blocs of three lines for 1 source positions
        Gsereega(:,:,:) = repmat(Gmne,1,1,3); 
        
        % doubt??? %
        %oris = source_orientations; % keeping the same orientation vector.
        oris = (1/3.0)*ones(3*nb_sources,3); 
        spos = zeros(nb_sources*3, 3); 
        % Also reshape the positions... %%% 
        i1 = 1:3:3*nb_sources; 
        spos(i1,:)   = source_positions; 
        spos(i1+1,:) = source_positions; 
        spos(i1+2,:) = source_positions; 
        
    end