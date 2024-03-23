function [Accuracy,time] = Wave_Adam_function(alltrain,test,a,b,C,mew,m,max_iter,beta1,beta2,alpha,epsilon,t)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Inputs of function
%       alltrain denotes the training data.
%       test denotes the test data.
%       a and b are wave loss parameters.
%       C and mew are regularization parameter and kernel parameter,
%       respectively.
%       beta1 and beta2 are exponential decay rates for the first and second moment estimate
%       m denotes the size of mini-batch.
%       max_iter denotes the number of maximum iteration.
%       alpha is the learning rate.
%       epsilon is a small constant used to avoid division by zero.
%       t denotes the iteration number.


%%     Output of function
%      Accuracy and time denotes the classification accuracy and
%      training time of the model.

l=size(alltrain,1);
rand_num=randperm(l);
rand_data=zeros(m,size(alltrain,2));

for i=1:m
    rand_data(i,:)=alltrain(rand_num(i),:);
end
%% xrand and yrand are the feature matrix and labels of m randomly selected training samples.
xrand=rand_data(:,1:end-1); yrand=rand_data(:,end);

%% Split the feature and label of the Test set
Xtest=test(:,1:end-1);
Ytest=test(:,end);

%% Generating the kernel matrix for the training data using m randomly selected training samples.
XX=sum(xrand.^2,2)*ones(1,m);
omega=XX+XX'-2*(xrand*xrand');   %omega is the kernel matrix for data X.
omega=exp(-omega./(2*mew^2));

% initialize the parameters
% n1=size(xrand,2); % feature in dataset
gamma=0.01*ones(m,1);  % initialize model parameter
r=0.01*ones(m,1);      % initialize first order moment
v=0.01*ones(m,1);      % initialize second order moment

%% finding xi_i

q=zeros(m,1);  %This is summation term in xi_i
for i=1:m
    q(i)=sum(gamma.*omega(:,i));
end

u=zeros(m,1);   % This is xi_i
for i=1:m
    u(i)=(yrand(i)*q(i))-1;
end

%% finding gradient
E=zeros(m,m);
for i=1:m
    E(i,:)=((u(i)*exp(a*u(i))*yrand(i)*(2+u(i)))/(1+b*u(i)*u(i)*exp(a*u(i)))^2)*omega(i,:);
end

tic
% Optimization loop

for i = 1:max_iter
    t = t + 1;
    gradient= omega*gamma - C*sum(E,1)';

    % Update bias-corrected first and second moment estimates
    r = beta1 .* r + (1 - beta1) .* gradient;
    v = (beta2 .* v) + ((1 - beta2) .* (gradient.^2));
    r_hat = r ./ (1 - beta1^t);
    v_hat = v ./ (1 - beta2^t);
    gamma = gamma - ((alpha .* r_hat) ./ (sqrt(v_hat) + epsilon));
end

% Return optimal solution and function value
gamma_opt = gamma;


XK=xrand; % Storing X in another matrix so that all the upgradation while calculating kernel will be done in new matrix

p=size(Xtest,1);
omega1=-2*XK*Xtest';
XK=sum(XK.^2,2)*ones(1,p);
Xtest=sum(Xtest.^2,2)*ones(1,m);
omega1=omega1+XK+Xtest';
omega1=exp(-omega1./2*mew^2); % omega1 is the kernel matrix corresponding to test data projected on training data

HT=omega1.*yrand;


f=sign(HT'*gamma_opt);
time=toc;

%% Finding Accuracy using true positive(tp), true negative(tn), false positive(fp) and false negative(fn)
tp=0;tn=0;fp=0;fn=0;
for j=1:length(Ytest)
    if Ytest(j)>0
        if Ytest(j)==f(j)
            tp=tp+1;
        else
            fn=fn+1;
        end
    end
    if Ytest(j)<0
        if Ytest(j)==f(j)
            tn=tn+1;
        else
            fp=fp+1;
        end
    end
end
Accuracy=(tp+tn)/(tp+fn+fp+tn);
end
