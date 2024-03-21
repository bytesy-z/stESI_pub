function order_ok = utl_check_max_order(csv_file, n_patch, order_max, margin)
    c = readcell(csv_file);
    
    idx = cell2mat( c(2:end,1) )==n_patch & cell2mat( c(2:end,2) )==margin; 
    legal_o_max = cell2mat( c( idx,3 ) );
    
    if legal_o_max < order_max
        order_ok = false; 
    else
        order_ok = true; 
end