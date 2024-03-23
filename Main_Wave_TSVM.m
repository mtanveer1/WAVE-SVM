close all;clear all;clc;

load("Train.txt");
load("Test.txt");
X_train=Train(:,1:end-1);
Y_train =Train(:,end);
X_test=Test(:,1:end-1);
Y_test =Test(:,end);


    %% default parameters
    C=1;   %  The structural risk term regularization parameter
    c=1;   % The loss term regularization parameter
    a= 1;  % a and l are the wave loss function parameters
    l= 1;
    
                            
   [uu1,uu2,bb1,bb2,Accuracy,time]=Wave_TSVM_function(X_train,Y_train,X_test,Y_test,a,l,C,c);

   disp(Accuracy);

                            
