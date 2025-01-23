% Numerical experiments for the derivatives of inertial methods 
% The measurements vectors are Gaussian random matrix

% n: the size of signal vectors
% m: the number of the measurements 
% L:the coefficients of Lipschitz continuity
% Testing differentiating
% Dependency=create_signal,create_Operator,getoptions,perform_autodiif_sol,
% perform_inert_method,x_bar_teta,perform_autodiif

clear all;
clc;

%Global Parameters of the  code		
        
n=128;              % size of the signal which to be recovered
m= floor(n*log(n)); % number of measurements 
niter=300;          % number of iterations
L=3;				% lipschitz continuity 
gamma=1/L;          % limit of the stepsize sequence can be taken ]0,2/L[                        
theta=10;           % the differentiation parameters depending on the values
dim=1;              % as usual the dimension of the signal     
SignalType = 'smooth rough'; % signal type can be gaussian, smooth rough or others

% Initilizing the parameters ak,bk and the stepsize gak 
ak=zeros(niter,1);
gak=zeros(niter,1);
for t=1:niter
    %ak(t)=(t-1)/(t+15);
    ak(t)=0;
    %gak(t)=gamma;
    gak(t)=1/(L-2/t);
    %ak(t)=sqrt(5)-2-1e-3;
end 
bk=ak;

% Options to define the function xbar(theta) (random_ square, smooth rough or random)
option_xbar.n=n;
option_xbar.dim=dim;
option_xbar.SignalType=SignalType;

% x=xbar(theta), xtilde
[x,xtilde]=x_bar_teta(theta,option_xbar); % Make signal (random_ square, smooth rough or random)

% Create operator for the problem

[A,At] = create_Operator(n,m,dim,'real gaussian'); 

%Generate  data measurements
Y = A(x);   

% Objective function and his derivatives
f= @(z) norm(A(z)-Y,2)^2/(2*m);         %objectives functions :least-square 
gradf= @(z)At(A(z)-Y)/m;                % gradient with respect to x
hesf_x=@(z) At(A(eye(n)))/m;            % hessian w.r.t x: $nabla^2_{x,x}f$
hesf_xt=@(z) -(At(A(xtilde.*theta))/m); % hessian $nabla^2_{x,theta}f$

%Initilization for the problem

zinit = randn(n,1);                 

% Function for recording the  errors of the iteration
% Can be generalized to take two parameters :recerror=@(z,x)norm(x-z, 'fro');
 
recerror_it = @(z)norm(x-z,'fro');  

% Computing pXsol using the explicit formula
% this function allows to compute the explicit 
% derivatives of Xsol with respect to the variable theta
% we denote it pXsol=$\partial_{\theta}\Xsol(theta)$


options_pXsol.n=n;          % adding the diemension of the problem can be avoided
options_pXsol.a=1;          % limit of the inertial parameter ak 
options_pXsol.b=1;          % limit of the inertial parameters bk
options_pXsol.gamma=gamma;  %  adding the limit of the stepsizes gak 

[pXsol,M]=perform_autodiif_sol(hesf_x(x),hesf_xt(x),options_pXsol); %perform the autodiff

%%%%---Largest eigenvalue of M------%%%%%%
rho_M=abs(eigs(M,1,'largestabs')); 

recerror_dif = @(z)norm(pXsol-z,'fro'); 



%options for the descent and the autodiff

options_im.niter = niter;             %number of iterations 
options_im.recerror_it = recerror_it; %record the error of the iteration
options_im.recerror_dif=recerror_dif; %record the error of the autodiff
options_im.hesf_x= hesf_x;            %enter the function of the hes w.r.t x
options_im.hesf_xt= hesf_xt;          %enter the function of the hes w.r.t x,teta
options_im.autodiff=1;                %autodiff parameters 0=noautodiff 1=autodiff 
options_im.pzinit=zeros(n,1);         %derivative of the Xinit called p
options_im.ak = ak;                   % adding the parameter ak  
options_im.bk = bk;                   % adding the parameter bk 
options_im.gak=gak;                   % adding the parameter gak
options_im.L  = L;                    % adding the Lipschitz coefficient L

% Performing the descent or the autodiff

% We just return the z_im the solution of the least-square 
% the error of inert method and the differentiation

[z_im,outerror_iter,outerror_diff]=perform_inert_method(zinit,gradf,options_im);




