function [uu1,uu2,bb1,bb2,Accuracy,time]=Wave_TSVM_function(X_train,Y_train,X_test,Y_test,a,l,C,c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input:
%    X_train: Training data features.
%    Y_train: Training data labels. 
%    X_test: test data features.
%    Y_test: Test data labels. 
% c: The loss term regularization parameter
% C: The structural risk term regularization parameter
% a and l: The wave loss function parameter 
 
%% Output:
% uu1: The positive hypersurface parameter u_+
% uu2: The negative hypersurface parameter u_-
% bb1: The positive hypersurface parameter b_+
% bb2: The negative hypersurface parameter b_-
% Accuracy and time denotes the classification accuracy and training time of the model.


X=X_train;
Y=Y_train;
A=[X,Y];
KAC=A(A(:,end)==1,1:end-1);  
KBC=A(A(:,end)==-1,1:end-1);


n1=size(KAC,2);
n2=size(KBC,2);
n3=size(KAC,1);
n4=size(KBC,1);

tic;
function f1_Z1=gfun1(Z1)
R2=zeros(size(KBC,2),size(KBC,1));
for j=1:size(KBC,1)

R2(:,j)=c.*KBC(j,:)' .* (   ( (1+KBC(j,:)*Z1) * (a*(1+KBC(j,:)*Z1)+2) * exp(a*(1+KBC(j,:)*Z1)) ) / ( 1 + l*((1+KBC(j,:)*Z1)^2)* exp(a*(1+KBC(j,:)*Z1)))^2    ) ;            
end
f1_Z1=sum(R2,2);
end



function f2_Z2=gfun2(Z2) 
S2=zeros(size(KAC,2),size(KAC,1));
for j=1:size(KAC,1)

S2(:,j)= -c.*KAC(j,:)'.* (   ( (1-KAC(j,:)*Z2) * (a*(1-KAC(j,:)*Z2)+2) * exp(a*(1-KAC(j,:)*Z2)) ) / ( 1 + l*((1-KAC(j,:)*Z2)^2)* exp(a*(1-KAC(j,:)*Z2)))^2    ) ;          
end
f2_Z2=sum(S2,2);
end
N=50;
t=1;
Z10=zeros(size(KAC,2),1);
Z20=zeros(size(KBC,2),1);

while(t<N)   
 g1t=feval(@gfun1,Z10);   
 Z1=1/C*(eye(n1)-KAC'*((C*eye(n3)+KAC*KAC')\KAC))*g1t;
    if(norm(Z1-Z10)<1e-6)
        break;
    end
    %f1=[f1,feval(@fun1,Z1)]; 
    Z10=Z1;
    t=t+1;
end
uu1=Z1(1:(size(Z1,1)-1),1);
bb1=Z1(end,1);
t=1;
while(t<N)
    g2t=feval(@gfun2,Z20);   
    Z2=-1/C*(eye(n2)-KBC'*((C*eye(n4)+KBC*KBC')\KBC))*g2t;
    if(norm(Z2-Z20)<1e-6)
        break;
    end
   
    Z20=Z2;
    t=t+1;
end
uu2=Z2(1:(size(Z2,1)-1),1);
bb2=Z2(end,1);
time=toc;
u1=[uu1;bb1];
u2=[uu2;bb2];
d1=abs(X_test*u1);
d2=abs(X_test*u2);
y=d1-d2;
y(y<0)=1;
y(y~=1)=-1;
preY=y;
err=sum(preY~=Y_test)/size(X_test,1);
Accuracy=1-err;

end


