close all;clear all;clc;

load("Train.txt");
load("Test.txt");



%% Set default values for parameters
beta1=0.9;       % exponential decay rates for the first moment estimate
beta2=0.999;     % exponential decay rates for the second moment estimate
alpha=0.01;      % learning rate
epsilon= 10^-6;  % small constant used to avoid division by zero
max_iter = 1000;  % maximum iteration number
t=0;
m=2^5;           % mini batch size
a=1.5;             % a and b are loss parameter
b=1;
C=1;           % Regularization parameter
mew=1;         % kernel parameter




[Accuracy,time] = Wave_Adam_function(Train,Test,a,b,C,mew,m,max_iter,beta1,beta2,alpha,epsilon,t);


disp(Accuracy);