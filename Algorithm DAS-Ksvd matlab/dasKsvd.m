function Phi_D = dasKsvd(data,labels,varargin)

% Inputs:
%       - data in an m x n matrix where m is the dimension of each signal
%              and n is the number of signals.   

%% parse inputs
parser = inputParser;

addParamValue(parser,'q',0.2);                                  %sparsity
addParamValue(parser,'r',1);                                    %redundancy factor
addParamValue(parser,'p_percent',0.2);                          %samples per class to train dictionary at each iteration (in percent of total of the less frequent class)
addParamValue(parser,'I',50);                                   %number of SD-KSVD iterations
addParamValue(parser,'beta',0.5);                               %bootsrtap parameter
addParamValue(parser,'disc_function','comb_disc_measure');      %discriminative measure used
addParamValue(parser,'discMeasureParams',[1 0]);                %discriminative measure parameters (default value, [alpha beta] = [1 0])
addParamValue(parser,'KSVDIter',20);                            %number of KSVD iterations
addParamValue(parser,'grid_iter',1)

parse(parser,varargin{:});


%% Initialize
[N, n] = size(data);
classes = unique(labels);   % number of classes
k = length(classes);

discAtomIndices = zeros(1,k);

nclass = zeros(1,k);
p_dist_in = cell(1,k);
for classIndex = 1:k
    nclass(classIndex) = sum(labels==classes(classIndex));
    p_dist_in{classIndex} = ones(nclass(classIndex),1)/nclass(classIndex);
end

p = ceil(min(nclass)*parser.Results.p_percent);    % defines number of samples per class for KSVD
index_out_class = zeros(p,classIndex);
index_out = zeros(1,p*k);

n_g_comb = size(parser.Results.discMeasureParams,1); % number of possible combinations (grid search)
    
Phi_D = cell(1,n_g_comb);

%% KSVD algorithm settings
verbose = 'tr'; % specifies iteration number, number of replaced atoms, and target function value
params.Tdata = floor(N*parser.Results.q); 
params.dictsize = floor(N*parser.Results.r);
params.iternum = parser.Results.KSVDIter; % KSVD iterations
params.memusage = 'high';
msgdelta = 0;

%% OMP algorithm settings
ompfunc = @omp;

%% main loop

for iteration = 1:parser.Results.I
    
    for classIndex = 1:k          %recorre clases para selleccionar ejemplos para el ksvd de la iteracion iteration
        [ index_out_class(:,classIndex),p_dist_in{classIndex} ] = ...
            SDKSVD_get_indices(p, p_dist_in{classIndex}, parser.Results.beta);
        
        aux = find(labels==classes(classIndex));
        index_out(1+(classIndex-1)*p:classIndex*p) = aux(index_out_class(:,classIndex));       
    end
    
    % learn dictionary
    params.data = data(:,index_out); 
    if iteration>1 % add noise to generalize
        params.data = params.data+0.02*iteration*randn(N,k*p);
    end
    %
    params.initdict = get_initial_dict(params.data,parser.Results.r,0); % (data)
    fprintf('SD-KSVD number of iteration: %d \n',iteration)
    Phi = ksvd(params,verbose);
    
    % sparse coding
    A = full(ompfunc(Phi,params.data,[],ceil(parser.Results.q*N)));            % sparse matrix
    
    % apply discriminative criterion
    [m_comb, ell_j_plus, ell_j_star] = feval(parser.Results.disc_function, parser.Results.discMeasureParams, params.data, Phi, A, labels(index_out));               % evaluate discriminative measure
     
    for i = 1:n_g_comb
        %Phi_D{i} = zeros(N,parser.Results.I*k); % structured dictionary assembling
        %Revisar, probablemente la linea anterior va por fuera del for
        % O simplemente define solo el tamaño de Phi_D
        for classIndex = 1:k % for each class
            aux1 = zeros(1,params.dictsize); % for the firt class
            aux2 = find(ell_j_plus{i}==classIndex-1); % for the second class
            aux1(aux2) = m_comb{i}(aux2);
            [~, discAtomIndices(classIndex)] = max(aux1);
        end % end class structured
        Phi_D{i}(:,k*(iteration-1)+1:k*iteration) = Phi(:,discAtomIndices);
    end
    
    %save(strcat('saves/comb_measure_iter_',num2str(iteration),'.mat'),'Phi','m_comb','ell_j_plus','ell_j_star','Phi_D');
    
    %plot(Phi(:,discAtomIndices))
    
    %Phi_D(:,k*(iteration-1)+1:k*iteration) = Phi(:,discAtomIndices);
    
end
end
