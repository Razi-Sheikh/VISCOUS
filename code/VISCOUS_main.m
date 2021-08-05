function [ GMCM_out ] = VISCOUS_main(response, input)
%<><><><><><<><><><><><main file for VISCOUS algorithm<><><><><><<><><><>><
% Version V0-2021
% The algorithm is proposed by Sheikholeslami et al. (2021)
%----------------------------------------------------------------
% Programmed by Razi Sheikholeslami, University of Oxford
% E-mail: razi.sheikholeslami@gmail.com
% ----------------------------------------------------------------
% Original paper:
% Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) 
% VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes.
% Water Resources Research, 57, e2020WR028435. https://doi.org/10.1029/2020WR028435
% =============================================================== %
% Inputs:
%         response ==========>>> model output (n x 1)
%            input ==========>>> sample of input variables (n x m)

% where n = samples size; m = number of input variabples (e.g. model parameters, etc.)

% Output:
%          GMCM_out ==========>>> variance-based senstivity indices (1 x m)

% ================== Generate synthetic data =============================
nv = size(input,2); % nv is the number of random input variables;
x = [input, response];
[nv, index_GSA] = VISCOUS_input(nv);
order = 0;
Beta = lsq_data(x, order, index_GSA);

% ===================== GSA using Gaussian mixture copula =================
[GMCM_out] = GMC_GSA(x,index_GSA,nv,order,Beta);
save results
end
%% ************************************************************************
%  ***********                  Sub-functions                   ***********
%  ******                                                            ******
%  **                                                                    **
%% ==============================Sub-function1=============================
function [nv,index_GSA]=VISCOUS_input(nv)
for d = 1:nv
index_GSA{1,d} = d; %indices for which the variance-based sensitivity indices should be estimated
% index_GSA = {[1],[2],[3],[1,2],[1,3],[2,3],...}; for group of parameters
% and total effect
end
end
%% ==============================Sub-function2=============================
function Beta = lsq_data(sample,order,index)
% Least square regression of the trend function;
y = sample(:,end); % response;
nn = length(index); % Number of nv;
Beta = {};

for jj = 1:nn
    beta_temp = [];
    if order==1
        xsample = [];
        xsample = [ones(length(y),1),sample(:,index{jj})]; % Samples of X;
        beta_temp = (xsample'*xsample)^(-1)*xsample'*y;
    elseif order==0
        beta_temp = mean(y);
    end
    Beta{jj} = beta_temp;
end
end
%% ==============================Sub-function3=============================
function SA_indx = GMC_GSA(sample_org,index_GSA,nv,order,Beta)
usample = []; %cdfs;
x_Normspace = []; %Transformed data in the standard normal space;
for i = 1:nv+1
    usample(:,i) = ksdensity(sample_org(:,i),sample_org(:,i),'function','cdf');
    x_Normspace(:,i) = norminv(usample(:,i),0,1);
end

N_int = 1e4; %Sample size for numerical integration;
Var_output = var(sample_org(:,end)); %Var of response variable;
[~,num_GSA] = size(index_GSA); %indices for which the variance-based sensitivity indices should be estimated
N_MC = 1e4; %Sample size for numerical integration;
for id = 1:num_GSA
    x = [];
    x = sample_org;
    z_LSF = lsq_pred(sample_org(:,index_GSA{id}),order,Beta{id});
    Y_res = sample_org(:,end) - z_LSF; %residuals;
    x(:,end) = Y_res; %Update the last colum of the data;  
    u_sample = ksdensity(Y_res,Y_res,'function','cdf');
    x_Normspace(:,end) = norminv(u_sample,0,1); % Update x in standard normal space;
    
    % Samples of residual for integration;
    U_MC = 0.0001:(0.999-0.0001)/(N_int-1):0.999;
    Y_int = ksdensity(x(:,end),U_MC','function','icdf');
    Y_Normspace = norminv(U_MC',0,1);
    
    %
    x_Normspace_first = x_Normspace(:,index_GSA{id}); %variables of interest;
    x_Normspace_all_first = [x_Normspace_first, x_Normspace(:,end)]; %input-output matrix;
    OptimalModel_first_all = GMCM_fit(x_Normspace_all_first,length(index_GSA{id})+1);
%     u = usample(:,id); 
%     v = usample(:,end);
%     Cuv(:,id) = cdf(BestModel_first_all,[u v]);
%     uv(:,id) = u.*v;  
    GMCM_based_samples = [];
    GMCM_X_int = []; 
    GMCM_based_samples = random(OptimalModel_first_all,N_MC); %samples from GMCM;
    GMCM_X_int = GMCM_based_samples(:,1:length(index_GSA{id})); 
    % Transform back into original space;
    cdf_int = normcdf(GMCM_X_int,0,1);
    GMCM_X_org = [];
    for i = 1:length(index_GSA{id})
       GMCM_temp = ksdensity(sample_org(:,index_GSA{id}(i)),cdf_int(:,i),'function','icdf'); 
       GMCM_X_org = [GMCM_X_org,GMCM_temp];
    end
    
    %Estimating conditional mean and variance;
    cond_Mean = [];
    Var_con = [];
    for i = 1:N_MC
       x_temp0 = GMCM_X_org(i,:);
       x_temp1 = GMCM_X_int(i,:);
       y_lsq_pred = lsq_pred(x_temp0,order,Beta{id}); 
       x_all_temp = [ones(N_int,1)*x_temp1,Y_Normspace]; 
       PDF_all_one = pdf(OptimalModel_first_all, x_all_temp);
       PDF_all_two = 1./normpdf(Y_Normspace,0,1);
       PDF_con = pdf_estimation(OptimalModel_first_all,x_all_temp);
      % pdf_con = pdf(BestModel_first_con, xtemp);
       cond_Mean(i) = (1/PDF_con/N_int)*sum(Y_int.*PDF_all_one.*PDF_all_two);
       %mu2_con = (1/PDF_con/N_int)*sum((Y_int.^2).*PDF_all_one.*PDF_all_two);
       %Var_con(i) = mu2_con-(mean_con(i))^2; % Variance of the residual--> Conditional variance;
       cond_Mean(i) = y_lsq_pred + cond_Mean(i); % Update the mean value by adding trend function term;
    end
    SA_indx(id) = var(cond_Mean)/Var_output;
end
end
%% ==============================Sub-function4=============================
function Z = lsq_pred(sample,order,beta)
% Prediction use least square regression of the trend function;
[n,~]=size(sample);
if order==1
    xsample = [];
    xsample = [ones(n,1),sample];
    Z = xsample*beta;
elseif order==0
    Z = beta*ones(length(sample(:,1)),1);
end
end
%% ==============================Sub-function5=============================
function Optimal_model = GMCM_fit(x,nv)
% Fit Gaussian mixture copulas
n = nv + 20;
BIC = zeros(1,n);
GMM_models = cell(1,n);
options = statset('MaxIter',2000);

for k = 1:n
    GMM_models{k} = fitgmdist(x,k,'Replicates',5,'RegularizationValue',...
        0.000001,'Options',options);
    BIC(k)= GMM_models{k}.BIC;
end

[~,numComponents] = min(BIC);
Optimal_model = GMM_models{numComponents};
end
%% ==============================Sub-function6=============================
function [conditioinal_pdf] = pdf_estimation(GMCM_model,u)
% Estimating conditional pdf using GMCM;
nn = length(GMCM_model.Sigma(1,:,1));
num_comp = GMCM_model.NComponents; 
pdf_con = [];

for kk = 1:num_comp
   mean_f = GMCM_model.mu(kk,:); %mean of the kk-th component;
   covar_f = reshape(GMCM_model.Sigma(:,:,kk),[nn,nn]); %covariance of the kk-th component;
%  PDF_component(:,i)=mvnpdf(U,Full_mu,Full_COV);
   pdf_con(kk) = mvnpdf(u(1,1:end-1),mean_f(1,1:end-1),covar_f(1:end-1,1:end-1));
end
conditioinal_pdf = sum(GMCM_model.PComponents.*pdf_con);
end
%%

















