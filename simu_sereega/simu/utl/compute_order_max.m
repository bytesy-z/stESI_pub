function o_max = compute_order_max( n_patch, margin, neighbors, order_max, save, plot)
    if nargin < 5
        save=true;
        plot = true;
    end
    if nargin > 5 && nargin < 4
        plot=true; 
    end
    n_sources = numel(fieldnames(neighbors)); n_sources = n_sources(1);
    
    c_save = {"seed", "order", "n_src_vois"}; 
    for seed = 1:n_sources
        for o = 1:order_max
            patch = utl_get_patch(o, seed, neighbors);
            n_source_patch = length(patch); 
            %pos = sPos(patch,:); 
            %dist = sqrt( ...
            %    (pos(:,1)-pos(:,1)').^2 + ...
            %    (pos(:,2)-pos(:,2)').^2 + ...
            %    (pos(:,3)-pos(:,3)').^2 );
            %mdist = max( dist, [], "all"); 
       
            c_save = [c_save; ...
                { seed, o, n_source_patch } ]; 
        end
    end
    if save
        writecell(c_save, strcat("stats_ordre_vois_ico3_test.csv") );
    end
    
    
    n_srcs = zeros(n_sources,order_max); 

    for o = 1:order_max
        srcs = cell2mat(c_save(2:end,3) ); 

        n_srcs(:,o) = srcs( cell2mat(c_save(2:end,2)) == o );
    end
    mus = mean(n_srcs);
    maxs = max(n_srcs, [], 1);
    res_mean = zeros(order_max,1);
    res_maxs = zeros(order_max,1);
    for o = 1:order_max
        res_mean(o) = floor( n_sources/mus(o) ); 
        res_maxs(o) = floor( n_sources/maxs(o) );
    end
    
    if plot
        figure()
        subplot(121)
        boxplot( n_srcs );
        xlabel("order"); ylabel("n sources"); 
        title("Boxplot of number of sources in a cluster for each cluster order")
        subplot(122)
        plot( res_mean, '.--' ); hold on 
        plot(res_maxs, '.--');
        legend("mean", "max");
        xlabel("order"); ylabel("n_sources/mean(n_source_patch):n patch max"); 
        title("~ number of non overlapping patches in source space")
    end
    
    o_max = max( find( res_maxs>n_patch+margin ) , [], "all") ; 
    % +margin to have a not only non overlapping patches but also some space between
    % patches

    disp(strcat("For up to ", num2str(n_patch), " patches, max order of patches = ", num2str(o_max) )) ; 
    disp(strcat("With a 'margin' of ", num2str(margin)) );
    
    if save
        writecell(c_save, strcat("max_order.csv") );
    end
    %% reflexion sphere



