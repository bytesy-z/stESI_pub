function [ampl, center, width] = erp_random_parameters(range_ampl, range_center, range_width)

    if range_ampl(2)>range_ampl(1) || range_ampl(1)>range_ampl(2)
        a1 = range_ampl(1); a2 = range_ampl(2); 
        ampl = (a2-a1)*rand(1)+a1; 
    else
        ampl = range_ampl(1); 
    end
    
    if range_center(2)>range_center(1) || range_center(1)>range_center(2)
        c1 = range_center(1); c2 = range_center(2); 
        center = (c2-c1)*rand(1)+c1; 
    else
        center = range_center(1);
    end
    
    if range_width(2)>range_width(1) || range_center(1)>range_center(2)
        w1 = range_width(1); w2 = range_width(2); 
        width = (w2-w1)*rand(1)+w1; 
    else
        width = range_width(1);
    end

end