function [m_comb, ell_j_plus, ell_j_star] = comb_disc_measure(params, data, Phi, A, labels, varargin)
%   year: 2018
%   Research institute for Signals, Systems and Computational Intelligence sinc(i)
%   http://www.sinc.santafe-conicet.gov.ar/
%   rrolon@sinc.unl.edu.ar
% Description:
% Parameters:
%   Phi: dictionary
%   A: sparse coefficient matrix
%   labels: class labels vector
%   par: weights (par = [alpha beta])
% Recommendation:
%   ...
% Caution:
%   alpha>=0, beta>=0, alpha+beta<=1 (convex combination)

n_parameters = size(params,1);
m_comb = cell(1,n_parameters);
ell_j_plus = cell(1,n_parameters);
ell_j_star = cell(1,n_parameters);

for i = 1:n_parameters
    alpha = params(i,1);
    beta = params(i,2);
    [m,n] = size(A);
    classes = unique(labels);   % classes  
    k = length(classes);        % number of classes
    act_prob = zeros(1,k);
    ell_j_plus{i} = zeros(1,m);
    ell_j_star{i} = zeros(1,m);
    p_ell_j_plus = zeros(1,m); q_ell_j_plus = zeros(1,m); r_ell_j_plus = zeros(1,m); 
    p_ell_j_star = zeros(1,m); q_ell_j_star = zeros(1,m); r_ell_j_star = zeros(1,m);
    m_af = zeros(1,m); % activation frequency
    m_cm = zeros(1,m); % coefficient magnitude
    m_re = zeros(1,m); % representation error
    %atoms_new = 1:m;
    % Incorporacion de calculo de error para mejora de velocidad en m_re
    errorinclass = cell(1,k);
    for cl = 1:k
        errorinclass {cl} = data(:,k)-Phi*A(:,k);
    end
    %---------------------
    % activation frequency
    %---------------------
    for atomIndex = 1:m % for each atom
        for classIndex = 0:(k-1)            % for each class
            labels_ell = find(labels==classes(classIndex+1));
            n_ell = length(labels_ell); % number of class ell samples
            act_prob(classIndex+1) = sum(A(atomIndex,labels_ell)~=0)/n_ell; % p_{\ell}^j see paper
        end
        [aux1,aux2] = sort(act_prob,'descend');
        % classes
        ell_j_plus{i}(atomIndex) = aux2(1)-1;
        ell_j_star{i}(atomIndex) = aux2(2)-1;
        % activation probabilities
        p_ell_j_plus(atomIndex) = aux1(1);
        p_ell_j_star(atomIndex) = aux1(2);
        % activation frequency measure
        m_af(atomIndex) = (p_ell_j_plus(atomIndex)-p_ell_j_star(atomIndex))/p_ell_j_plus(atomIndex);
    end
    for atomIndex = 1:m
        aux3 = find(labels==ell_j_plus{i}(atomIndex));
        aux4 = find(labels==ell_j_star{i}(atomIndex));
        n_ell_j_plus = length(aux3);
        n_ell_j_star = length(aux4);
        %---------------------
        % coefficient magnitude
        %---------------------
        q_ell_j_plus(atomIndex) = sum(abs(A(atomIndex,aux3)))/n_ell_j_plus;
        q_ell_j_star(atomIndex) = sum(abs(A(atomIndex,aux4)))/n_ell_j_star;
        if q_ell_j_plus(atomIndex)>q_ell_j_star(atomIndex)
            m_cm(atomIndex) = (q_ell_j_plus(atomIndex)-q_ell_j_star(atomIndex))/q_ell_j_plus(atomIndex);
        end
        %---------------------
        % representation error
        %---------------------
        % Modificacion para eficiencia de velocidad
        auxclass1 = ell_j_plus{i}(atomIndex)+1;
        auxclass2 = ell_j_star{i}(atomIndex)+1;
        r_ell_j_plus(atomIndex) = norm(errorinclass{auxclass1}+Phi(:,atomIndex)*A(atomIndex,auxclass1),'fro')/n_ell_j_plus;
        r_ell_j_star(atomIndex) = norm(errorinclass{auxclass2}+Phi(:,atomIndex)*A(atomIndex,auxclass2),'fro')/n_ell_j_star;
        %atoms_new(atoms_new==atomIndex) = [];
        %r_ell_j_plus(atomIndex) = norm(data(:,aux3)-Phi(:,atoms_new)*A(atoms_new,aux3),'fro')/n_ell_j_plus;
        %r_ell_j_star(atomIndex) = norm(data(:,aux4)-Phi(:,atoms_new)*A(atoms_new,aux4),'fro')/n_ell_j_star;
        if r_ell_j_plus(atomIndex)<r_ell_j_star(atomIndex)
            m_re(atomIndex) = (r_ell_j_star(atomIndex)-r_ell_j_plus(atomIndex))/r_ell_j_star(atomIndex);
        end
        %atoms_new = 1:m;
    end  
    m_comb{i} = alpha*m_af+beta*m_cm+(1-alpha-beta)*m_re;
    clear alpha beta
end



end