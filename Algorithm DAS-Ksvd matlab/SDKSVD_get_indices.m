function [ index_out,p_dist_out ] = SDKSVD_get_indices(p, p_dist_in, beta)
% Randomly selects p indexes using the probabilities given in p_dist_in.
% And updates p_dist_out using factor beta (0 =< beta < 1).

    p_dist_out = p_dist_in;

    index_out = randsample(1:size(p_dist_in),p,true,p_dist_in);
    %index_out = bootsmp(p_dist_in,p);
    auxdisp = p_dist_in;
    dif = p - length(unique(index_out) );
    while  dif > 0
        auxdisp(index_out) = 0;
        auxret = randsample(1:size(auxdisp),dif,true,auxdisp);      
        index_out = [unique(index_out), auxret];
        dif = p - length(unique(index_out) );
    end
    
    p_dist_out(index_out) = p_dist_in(index_out)*beta;
    p_dist_out = p_dist_out./sum(p_dist_out);

end