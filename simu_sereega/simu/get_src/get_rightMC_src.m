function [idx, pos] = get_rightMC_src(lf, viz)

    if nargin<2
        viz=false;
    end
    sPos = lf.pos; % possible source positions

    %(copy and paste from simBCI where_heuristic...]
    mx = (max(sPos(:,1))+min(sPos(:,1)))/2;
    my = (max(sPos(:,2))+min(sPos(:,2)))/2;
    mz = (max(sPos(:,3))+min(sPos(:,3)))/2;
    rightCenter =  [(mx+max(sPos(:,1)))/2,my,max(sPos(:,3))/2];
    s_rightMC  = lf_get_source_nearest(lf, rightCenter);

    % Visualisation
    idx = s_rightMC;  pos = sPos(idx,:);
    if viz
        plot_source_location(idx, lf);
        title('Right motor cortex sources visualization')
    end

end