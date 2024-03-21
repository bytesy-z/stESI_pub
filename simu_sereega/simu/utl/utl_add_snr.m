function [data, SNRres] = utl_add_snr(data1, data2, snr) 
% NE fonctionne pas... probleme pendant la division par ligne.



    
    s = size(data1); 
    alph = utl_alpha_snr(snr); 
    data = zeros(s); 
    if length(size(data1))>2
        nb_trials = s(3); 
        for i = 1:nb_trials
            tmp_data1 = data1(:,:,i); 
            tmp_data2 = data2(:,:,i); 
            D1 = tmp_data1/norm(tmp_data1,'fro');
            D2 = tmp_data2/norm(tmp_data2,'fro');
            data(:,:,i) = alph*D1 + (1-alph)*D2;
    %scale = max(abs(Msrc), [], 'all'); 
%     data = max(abs(data1),[],2).*data./max(abs(data),[],2);%
            data(:,:,i) = max(abs(tmp_data1),[],'all').*data(:,:,i)./max(abs(data(:,:,i)),[],'all');%
            
        end
        [SNRmeas, SNRmeas_db] = utl_compute_snr(D1(1,:), D2(1,:), 'rms');
        SNRres = [SNRmeas, SNRmeas_db];
    else 
        D1 = data1/norm(data1,'fro'); 
        D2 = data2/norm(data2,'fro');
        data = alph*D1 + (1-alph)*D2;
    %scale = max(abs(Msrc), [], 'all'); 
%     data = max(abs(data1),[],2).*data./max(abs(data),[],2);%
        data = max(abs(data1),[],'all').*data./max(abs(data),[],'all');%

        [SNRmeas, SNRmeas_db] = utl_compute_snr(D1(1,:), D2(1,:), 'rms');
        SNRres = [SNRmeas, SNRmeas_db];
    
        
    end

end