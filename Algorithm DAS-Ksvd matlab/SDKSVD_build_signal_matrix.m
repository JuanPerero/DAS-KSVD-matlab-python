function [signal, target] = SDKSVD_build_signal_matrix(spo2, apn_spo2, win, over)
%   frames = floor((length(spo2)-win)/over);
    frames = floor(length(spo2)/over)-2; 
    signal = zeros(win,frames);
    target = zeros(1,frames);
    j = 1;
    while j <= frames
        signal(:,j) = spo2(1+over*j:win+j*over);
        aux = apn_spo2(1+over*j:win+j*over);
    	if max(aux)<300
            if sum(aux==200) > sum(aux==100)
                target(j) = 200;
            elseif sum(aux==100)== 0
                target(j) = 0;
            else
                target(j) = 100;
            end
            %target(j) = max(aux)>0; Caso binario
      
        end
        j = j+1;
    end
end

