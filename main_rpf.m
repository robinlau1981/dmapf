%% This code is finished on Aug. 16th in 2016.
%% It implements a robust PF algorithm based on a dynamically weighted mixture model of the observation noise
%% The mixture components include two Student's t distributions and a Gaussian
%% This code is developed based on PF code written by Nando de Freitas and Rudolph van der Merwe

clear;
close all;
echo off;
% you should install the lightspeed toolbox at first.  See it at: https://github.com/tminka/lightspeed
addpath(genpath('F:\MATLAB\R2018a\work\Toolbox_plus\lightspeed')); 
% INITIALISATION AND PARAMETERS:
% ==============================
no_of_runs = 1;            % number of experiments to generate statistical
                            % averages
doPlot = 0;                 % 1 plot online. 0 = only plot at the end.
sigma =  1e-2;              % Variance of the Gaussian measurement noise.
g1 = 3;                     % Paramater of Gamma transition prior.
g2 = 2;                     % Parameter of Gamman transition prior.
T = 60;                     % Number of time steps.
Q = 3/4;                    % process noise variance.

N = 200;                    % Number of particles.
resamplingScheme = 1;       % The possible choices are
                            % systematic sampling (2),
                            % residual (1)
                            % and multinomial (3). 
                            % They're all O(N) algorithms. 

%**************************************************************************************
% SETUP BUFFERS TO STORE PERFORMANCE RESULTS
% ==========================================
rmsError_pf       = zeros(1,no_of_runs);
time_pf       = zeros(1,no_of_runs);  
%**************************************************************************************
% MAIN LOOP
df=3; % Degree of freedom of t 
df2=10;
clutter_min=20; 
clutter_max=30; % for generating clutters
M=3; % number of mixture components
ff=.9; % forgetting factor
ESS=zeros(no_of_runs,T);
pai_ave=zeros(T,M); % averaged pai over 30 Monte Carlo runs
for j=1:no_of_runs,    
    rand('state',sum(100*clock));   % Shuffle the pack!
    randn('state',sum(100*clock));   % Shuffle the pack!       
    % GENERATE THE DATA:
    % ==================    
    x = zeros(T,1);
    y = zeros(T,1);
    processNoise = zeros(T,1);
    measureNoise = zeros(T,1);
    pai=1/M*ones(1,M); % weights of mixture components
    pai_ave(1,:)=pai_ave(1,:)+pai;
    pai_tmp=pai;
    mLik=ones(1,M); % marginal likelihood of mixture components
    
    x(1) = 1;                         % Initial state.
    for t=2:T
        processNoise(t) = gengamma(g1,g2);        
        if (t>6 && t<10) || t==20 || (t>36 && t<40) || t==50                   % inject clutters
           measureNoise(t) = unifrnd(clutter_min,clutter_max);    
        else
           measureNoise(t) = sqrt(sigma)*randn(1,1); 
        end
        x(t) = feval('ffun',x(t-1),t) +processNoise(t);     % Gamma transition prior.  
        y(t) = feval('hfun',x(t),t) + measureNoise(t);      % Gaussian likelihood.       
    end  
    
    %%%%%%%%%%%%%%%  PERFORM SEQUENTIAL MONTE CARLO  %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%  ==============================  %%%%%%%%%%%%%%%%%%%%%
    
    % INITIALISATION:
    % ==============
    xparticle_pf = ones(T,N);        % These are the particles for the estimate
    % of x. Note that there's no need to store
    % them for all t. We're only doing this to
    % show you all the nice plots at the end.
    xparticlePred_pf = ones(T,N);    % One-step-ahead predicted values of the states.
    yPred_pf = ones(T,N);            % One-step-ahead predicted values of y.
    w = ones(T,N,M);                   % Importance weights.
    wm = ones(T,N);      % w after model averaging
    
    disp(' ');
    
    tic;                             % Initialize timer for benchmarking
    
    for t=2:T   
        fprintf('run = %i / %i :  PF : t = %i / %i  \r',j,no_of_runs,t,T);
        fprintf('\n')        
        
        % PREDICTION STEP:
        % ================ 
        % We use the transition prior as proposal.
        for i=1:N,
            xparticlePred_pf(t,i) = feval('ffun',xparticle_pf(t-1,i),t) + gengamma(g1,g2);
        end;
        
        % EVALUATE IMPORTANCE WEIGHTS:
        % ============================
        % For our choice of proposal, the importance weights are give by:  
        for i=1:N
            yPred_pf(t,i) = feval('hfun',xparticlePred_pf(t,i),t);    
            w(t,i,1) = normpdf(y(t)-yPred_pf(t,i),0,sqrt(sigma)) + 1e-99;
            w(t,i,2) = exp(log_t_pdf(y(t)-yPred_pf(t,i),0,sigma,df)) + 1e-99; % Deal with ill-conditioning. 
            w(t,i,3) = exp(log_t_pdf(y(t)-yPred_pf(t,i),0,sigma,df2)) + 1e-99; % Deal with ill-conditioning. 
        end 
        for i=1:M
            mLik(i)=sum(w(t,:,i)); 
        end
        
        pai_tmp=pai.^ff;
        pai_pred=pai_tmp/sum(pai_tmp); 
        pai=pai_pred.*mLik/sum(pai_pred.*mLik);
        
        for i=1:M
            w(t,:,i) = w(t,:,i)./sum(w(t,:,i));  % Normalise the weights.
        end  
        wm(t,:) =  pai(1)*w(t,:,1)+pai(2)*w(t,:,2)+pai(3)*w(t,:,3); 
        
        if 0% t==20 ||  t==50
           figure, plot(wm(t,:));
        end
        
        pai=max(pai,1e-1*ones(1,M)); % To guarantee that the minimum weight component still survives
        pai=pai/sum(pai);
        pai_ave(t,:)=pai_ave(t,:)+pai;
        %ESS(j,t)=1/sum(w(t,:).^2)/N;
        % SELECTION STEP:
        % ===============
        % Here, we give you the choice to try three different types of
        % resampling algorithms. Note that the code for these algorithms
        % applies to any problem!
        if resamplingScheme == 1
            outIndex = residualR(1:N,wm(t,:)');        % Residual resampling.
        elseif resamplingScheme == 2
            outIndex = systematicR(1:N,wm(t,:)');      % Systematic resampling.
        else  
            outIndex = multinomialR(1:N,wm(t,:)');     % Multinomial resampling.  
        end
        xparticle_pf(t,:) = xparticlePred_pf(t,outIndex); % Keep particles with
        % resampled indices.
    end;   % End of t loop.
    
    time_pf(j) = toc;    % How long did this take?
    
    %%%%%%%%%%%%%%%%%%%%%  PLOT THE RESULTS  %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%  ================  %%%%%%%%%%%%%%%%%%%%%
    
%     figure(2)
%     clf;
%     p0=plot(1:T,y,'k+','lineWidth',2); hold on;    
%     p4=plot(1:T,mean(xparticle_pf(:,:)'),'g','lineWidth',2);   
%     p1=plot(1:T,x,'k:o','lineWidth',2); hold off;
%     legend([p0 p1 p4],'Noisy observations','True x','PF estimate');
%     xlabel('Time','fontsize',15)
%     zoom on;
%     title('Filter estimates (posterior means) vs. True state','fontsize',15)
    
    if (0),
        figure(3),
        clf;
        % Plot predictive distribution of y:
        subplot(211);
        domain = zeros(T,1);
        range = zeros(T,1);
        thex=[-3:.1:15];
        hold on
        ylabel('Time (t)','fontsize',15)
        xlabel('y_t','fontsize',15)
        zlabel('p(y_t|y_{t-1})','fontsize',15)
        title('Particle Filter','fontsize',15);
        %v=[0 1];
        %caxis(v);
        for t=6:5:T
            [range,domain]=hist(yPred_pf(t,:),thex);
            waterfall(domain,t,range/sum(range));
        end
        view(-30,80);
        rotate3d on;
        a=get(gca);
        set(gca,'ygrid','off');
        % Plot posterior distribution of x:
        subplot(212);
        domain = zeros(T,1);
        range = zeros(T,1);
        thex=[0:.1:10];
        hold on
        ylabel('Time (t)','fontsize',15)
        xlabel('x_t','fontsize',15)
        zlabel('p(x_t|y_t)','fontsize',15)
        %v=[0 1];
        %caxis(v);
        for t=6:5:T
            [range,domain]=hist(xparticle_pf(t,:),thex);
            waterfall(domain,t,range/sum(range));
        end
        view(-30,80);
        rotate3d on;
        a=get(gca);
        set(gca,'ygrid','off');   
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %-- CALCULATE PERFORMANCE --%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    rmsError_pf(j)      = sqrt(inv(T)*sum((x'-mean(xparticle_pf')).^(2))); 
    Error_pf(j,:) = x'-mean(xparticle_pf');
    
    disp(' ');
    disp('Root mean square (RMS) errors');
    disp('-----------------------------');
    disp(' ');    
    disp(['PF           = ' num2str(rmsError_pf(j))]);  
    disp(' ');
    disp(' ');
    disp('Execution time  (seconds)');
    disp('-------------------------');
    disp(' ');
    disp(['PF           = ' num2str(time_pf(j))]);   
    disp(' ');    
    drawnow;
    
    %*************************************************************************

end    % Main loop (for j...)
%figure,plot(mean(ESS));
pai_ave=pai_ave/no_of_runs;
figure(6),plot(1:T,pai_ave(:,1)',1:T,pai_ave(:,2)',1:T,pai_ave(:,3)');grid on;
legend('Gaussian model','Student model (v=3)','Student model (v=10)');
xlabel('time');ylabel('posterior probability');
% calculate mean of RMSE errors
mean_RMSE_pf      = mean(rmsError_pf);
% calculate variance of RMSE errors
var_RMSE_pf      = var(rmsError_pf);
% calculate mean of execution time
mean_time_pf      = mean(time_pf);

% display final results

disp(' ');
disp(' ');
disp('************* FINAL RESULTS *****************');
disp(' ');
disp('RMSE : mean and variance');
disp('---------');
disp(' ');
disp(['PF           = ' num2str(mean_RMSE_pf) ' (' num2str(var_RMSE_pf) ')']);

disp(' ');
disp(' ');
disp('Execution time  (seconds)');
disp('-------------------------');
disp(' ');
disp(['PF           = ' num2str(mean_time_pf)]);
disp(' ');

%*************************************************************************

figure(5)
for j=1:no_of_runs
    plot(1:T, Error_pf(j,:),'LineWidth',1,'Color',[.4 .4 .4]);
    grid on; xlabel('Time');ylabel('Mean of error'); grid on;hold on;
end
%save Res_pf_mix_t_gauss_1D_ff0125 N ff mean_RMSE_pf var_RMSE_pf mean_time_pf df df2 M no_of_runs Error_pf;