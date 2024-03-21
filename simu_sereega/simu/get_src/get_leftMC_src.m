function [idx, pos] = get_leftMC_src(lf, viz)

    if nargin<2
        viz=false;
    end
    sPos = lf.pos; % possible source positions

    %(copy and paste from simBCI where_heuristic...]
    mx = (max(sPos(:,1))+min(sPos(:,1)))/2;
    my = (max(sPos(:,2))+min(sPos(:,2)))/2;
    mz = (max(sPos(:,3))+min(sPos(:,3)))/2;
    leftCenter =  [(mx+min(sPos(:,1)))/2,my,max(sPos(:,3))/2];
    s_leftMC  = lf_get_source_nearest(lf, leftCenter);

    % Visualisation
    idx = s_leftMC;  pos = sPos(idx,:);
    if viz
        plot_source_location(idx, lf);
        title('Left motor cortex sources visualization')
    end

end