function pdf =log_t_pdf(X,Mu,Sigma,df) %Ref: Robust mixture modelling using the t distribution;ML estimation of the t distribution using EM and its extensions,ECM and ECME
% X and Mu are row vectors
%The input to logdet must be a symmetric positive definite matrix. 
[n,p]=size(X);
% pdf=zeros(n,1);
% for k=1:n
%     %if rcond(Sigma)>1e-16
%         pdf(k)=log(gamma((df+p)/2))-.5*logdet(Sigma)-(.5*p*log(pi*df)+log(gamma(df/2))+.5*(df+p)*log(1+((X(k,:)-Mu)/Sigma*(X(k,:)-Mu)')/df));
%         
%     %elseif X(k,:)==Mu
% %            pdf(k)=0;
% %     %else
% %            pdf(k)=-Inf;
% %     end
% end
pdf=log(gamma((df+p)/2))-.5*logdet(Sigma)-(.5*p*log(pi*df)+log(gamma(df/2))+.5*(df+p)*log(1+sqdist(X',Mu',inv_posdef(Sigma))/df));
pdf=real(pdf);
