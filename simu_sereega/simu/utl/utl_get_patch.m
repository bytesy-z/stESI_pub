% function patch_idx = utl_get_patch(order, seed_idx, neighbors)
% 2022.07.15
% function to get all the index of sources in a patch of order "order", 
% around seed_idx.
% 
% input: 
% - order:  order of the patch to get (0 = single source).
% - seed_idx : index of patch seed
% - neighbors : structure containing the neighbors index of each source of
% the mesh, computed from utl_compute_neighbors.
% ouput: 
% - patch_idx : indices of the sources in the patch (unique index, contain
% also the source seed)
% 
% From: https://github.com/LukeTheHecker/esinet/blob/main/esinet/util/util.py
% Hecker, L., Rupprecht, R., Tebartz Van Elst, L., & Kornmeier, J. (2021). 
% ConvDip: A Convolutional Neural Network for Better EEG Source Imaging. 

% function patch_idx = utl_get_patch(order, seed_idx, neighbors)
%     f_names     = fieldnames(neighbors); 
%     patch_idx   = seed_idx;
%     if order ~= 0
%         for k = 1:order
%             for i = 1:length(patch_idx)
%                 ii = patch_idx(i);
%                 patch_idx = unique( [patch_idx, neighbors.( string(f_names(ii)) ) ] );
%             end
%         end
%         %patch_idx = unique(patch_idx);
%     end
% end

function patch_idx = utl_get_patch(order, seed_idx, neighbors)
    f_names     = fieldnames(neighbors); 
    
    patch_idx     = seed_idx;
    
    if order ~= 0
        for k = 1:order
            new_neighbs = []; 
            for i = 1:length(patch_idx)
                ii = patch_idx(i);
                new_neighbs = [new_neighbs, neighbors.( string(f_names(ii)) )' ];
            end

            patch_idx    = unique([ patch_idx, new_neighbs ]); 
        end
        
        patch_idx = unique(patch_idx);
    end
end

