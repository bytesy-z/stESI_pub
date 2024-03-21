function [eb_data, tvec] = sim_eyeBlink( fs, tvec)


    std_xsi = 20/100; 
    std_eb  = 31/1000; 
    
    xsi = rand(1)*std_xsi;
    
    %tvec_bis = linspace( 0, std_eb, std_eb*fs );
    %blink_start = ceil(  (fs*tvec(end)- length(tvec_bis) )*  rand(1) ); 
        
    %eb_data = zeros(size(tvec));
    
    %eb_data( blink_start:blink_start+length(tvec_bis)-1 ) = ...
    %    (tvec_bis+xsi).*exp( -tvec_bis.^2/(2*std_eb^2) ); 
    eb_data = (tvec+xsi).*exp( -tvec.^2/(2*std_eb^2) ); 


end