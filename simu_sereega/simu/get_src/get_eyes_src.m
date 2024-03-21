function [eyes_idx, eyes_pos] = get_eye_src(lf, viz)

    if nargin<2
        viz = false; 
    end
    
    sPos = lf.pos; 
    mx = (max(sPos(:,1))+min(sPos(:,1)))/2;
    my = (max(sPos(:,2))+min(sPos(:,2)))/2;
    mz = (max(sPos(:,3))+min(sPos(:,3)))/2;


    leftEye0 =  [(mx+min(sPos(:,1)))/2.2,max(sPos(:,2)),mz];
    leftEye = lf_get_source_nearest(lf, leftEye0); 
    rightEye0 = [(mx+max(sPos(:,1)))/2.2,max(sPos(:,2)),mz];
    rightEye = lf_get_source_nearest(lf, rightEye0); 

    eyes_idx = [leftEye, rightEye];
    eyes_pos = sPos(eyes_idx, :);
    if viz
        plot_source_location(idx, lf);
        title('Left motor cortex sources visualization')
    end


end
