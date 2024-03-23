# Wave-SVM - Advancing Supervised Learning with the Wave Loss Function: A Robust and Smooth Approach


This code corresponds to the paper Mushir Akhtar, M. Tanveer, and Mohd. Arshad. "Advancing Supervised Learning with the Wave Loss Function: A Robust and Smooth Approach".

If you are using our code, please give proper citation to the above given paper.

If there is any issue/bug in the code please write to phd2101241004@iiti.ac.in or im.mushir.akh@gmail.com.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Wave-SVM 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Wave-SVM code requires two files namely Wave_Adam_function.m and Main_Wave_SVM.m
%%  The descricption of files are given below:
   
1. Wave_Adam_function.m: This file contains the function of Adam algorithm utilized to solve the Wave-SVM. In Wave_Adam_function the inputs and their meanings are as follows:
   %       alltrain denotes the training data.
   %       test denotes the test data.
   %       a and b are wave loss parameters.
   %       C and mew are regularization parameter and kernel parameter,respectively.
   %       beta1 and beta2 are exponential decay rates for the first and second moment estimate
   %       m denotes the size of mini-batch.
   %       max_iter denotes the number of maximum iteration.
   %       alpha is the learning rate.
   %       epsilon is a small constant used to avoid division by zero.
   %       t denotes the iteration number.

   The outputs of Wave_Adam_function and their meaning are as follows:
   %       Accuracy and time denotes the classification accuracy and training time of the model.

   

2. Main_Wave_SVM.m: This is the main file of Wave-SVM. To utilize this code, you simply need to import the data and execute this script. Within the script, you will be required to provide values for various parameters (such as loss function parameters, Adam algorithm parameters, trade-off parameter, kernel parameter etc.).
To replicate the results achieved with Waev-SVM, you should adhere to the same instructions outlined in the paper "Advancing Supervised Learning with the Wave Loss Function: A Robust and Smooth Approach". 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Wave-TSVM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Wave-TVM code requires two files namely Wave_TSVM_function.m and Main_Wave_TSVM.m
%%  The descricption of files are given below:

1. Wave_TSVM_function.m: This file contains the function of iterative algorithm utilized to solve the Wave-TSVM. In Wave_TSVM_function the inputs and their meanings are as follows:
   %    X_train: Training data features.
   %    Y_train: Training data labels. 
   %    X_test: test data features.
   %    Y_test: Test data labels. 
   % c: The loss term regularization parameter
   % C: The structural risk term regularization parameter
   % a and l: The wave loss function parameter 

   The outputs of Wave_TSVM_function and their meaning are as follows:
  % uu1: The positive hypersurface parameter u_+
  % uu2: The negative hypersurface parameter u_-
  % bb1: The positive hypersurface parameter b_+
  % bb2: The negative hypersurface parameter b_-
  % Accuracy and time denotes the classification accuracy and training time of the model.

   

2. Main_Wave_TSVM.m: This is the main file of Wave-TSVM. To utilize this code, you simply need to import the data and execute this script. Within the script, you will be required to provide values for various parameters (such as loss function parameters, iteartive algorithm parameters, regularization parameters etc.).
To replicate the results achieved with Waev-TSVM, you should adhere to the same instructions outlined in the paper "Advancing Supervised Learning with the Wave Loss Function: A Robust and Smooth Approach". 





  

