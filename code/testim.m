% Numerical experiments for the derivatives of inertial methods 
% The measurements vectors are Gaussian random matrix
%n: the size of signal vectors
%m: the number of the measurements 
%L:the coefficients of Lipschitz continuity
%Testing least square before derivatives

clear all;
clc;

%Global Parameters of the  code															    															                % Number of sample
n=128;                   %dimension of the signal 
m= 3*floor(n*log(n)); 	 %number of measurements
niter=500;               %number of iterations
L=3;					 %Lipschitz continuity 

% Make signal 
dim=1;
SignalType = 'smooth rough';
x = create_signal(n,SignalType,dim);
% Create operator for the problem
[A,At] = create_Operator(n,m,dim,'real gaussian');
%Generate  data measurements
Y = A(x);
% Objective function, entropy and their derivatives
f= @(z) norm(A(z)-Y,2)^2/(2*m);
gradf= @(z)At(A(z)-Y)/m;
%Initilization for the problem
zinit = randn(n,1);
% Function for recording the  errors
recerror = @(z)norm(x-z, 'fro');
% Performing the descent
options_im.niter = niter;
options_im.recerror_it = recerror;
ak=zeros(niter,1);
gak=zeros(niter,1);
for t=1:niter
    ak(t)=(t-1)/(t+20);
    gak(t)=1/(L-2/t);
%    ak(t)=sqrt(5)-2-1e-3;
end 
bk=ak;
options_im.ak = ak;
options_im.bk = bk;
options_im.L  = L;
options_im.gak=gak;
[z_im,outerror_iter,outerror_diff]=perform_inert_method(zinit,gradf,options_im);