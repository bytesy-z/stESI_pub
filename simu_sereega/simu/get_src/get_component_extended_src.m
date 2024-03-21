% using a the extension based on neighbors from the mesh
% july 2022

function [c, patch, patch_dim] = get_component_extended_src(order, seed_idx, neighbors, sPos,...
    erp_params, erp_dev, LFsereega, timeline )
    
    amplitude = erp_params.ampl;
    % get indices of sources in patch
    patch               = utl_get_patch(order, seed_idx, neighbors); 
    n_sources_in_patch  = length(patch); 
    if order > 0
        seed_pos    = sPos(seed_idx, :);
        
        D_in_patch  = sqrt( sum( (repmat(seed_pos, n_sources_in_patch, 1) - sPos(patch,:)).^2, 2) );
        % patch dimension: 
        patch_dim = max(D_in_patch, [], 'all');
        D_in_patch  = D_in_patch./max(D_in_patch,[],'all'); %normalize

        sig     = max(D_in_patch, [], 'all')/2;
        ampl    = amplitude*exp( -0.5*(D_in_patch/sig).^2 );%./(sqrt(2*pi)*sig); 


        c = [];

        for s = 1:n_sources_in_patch
            a = ampl(s);  
            erp_params.ampl = a;
            tmp_c = get_component_Grech(LFsereega, patch(s),...
                timeline, erp_params, erp_dev, false);

            c = [c, tmp_c]; 
        end
    else
        c = get_component_Grech(LFsereega, patch,...
                timeline, erp_params, erp_dev, false);
    end
    
    
end%function
