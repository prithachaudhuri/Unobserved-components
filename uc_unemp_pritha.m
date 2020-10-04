%% Unobserved components model for US unemployment
% Code adopted from Josh Chan
% Author: Pritha Chaudhuri
% Date: 06/18/2020

clear
clc
close all
rng('default');

nsim = 2000;
burnin = 1000;
load('USUNEMP_2020Q1.mat'); % 1948:Q1-2020:Q4
y = USUNEMP2020Q1(1:272); % 1948Q1-2015Q4
T = length(y);

% intialize for storage
store_tau = zeros(nsim, T);
store_theta = zeros(nsim, 5); % [phi, sigc2, sigtau2, tau0], phi = [phi1, phi2]

% priors
a0 = 5;
b0 = 100;
nu_sigc2 = 3;
S_sigc2 = 1*(nu_sigc2-1);
nu_sigtau2 = 3;
S_sigtau2 = (0.25^2)*(nu_sigtau2-1);
phi0 = [1.6 -0.7]';
iVphi = speye(2);

% initialize Markov chain
sigc2 = 0.5;
sigtau2 = 0.01;
tau0 = 5;
phi = [1.66 -0.7]';

% compute matrix H
H = speye(T) - sparse(2:T, 1:(T-1), ones(1,T-1), T, T);
HH = H'*H;
HHiota = HH*ones(T,1);

% compute matrix Hphi
Hphi = speye(T) - phi(1)*sparse(2:T, 1:(T-1), ones(1,T-1), T, T) ...
    - phi(2)*sparse(3:T, 1:(T-2), ones(1,T-2), T, T);
count_phi = 0;

for isim = 1:(nsim+burnin)
    % step 1 : sample tau
    Ktau = HH/sigtau2 + (Hphi'*Hphi)/sigc2;
    tau_hat = Ktau\((tau0/sigtau2)*HHiota + (Hphi'*Hphi*y)/sigc2);
    Ctau = chol(Ktau, 'lower');
    tau = tau_hat + Ctau'\randn(T,1);
    
    % step 2: sample phi
    c = y - tau;
    Xphi = [[0;c(1:T-1)] [0;0;c(1:T-2)]];
    Kphi = iVphi + (Xphi'*Xphi)/sigc2;
    phi_hat = Kphi\(iVphi*phi0 + (Xphi'*c)/sigc2);
    Cphi = chol(Kphi, 'lower');
    phic = phi_hat + Cphi'\randn(2,1);
    if sum(phic) < 0.99 && phic(2)-phic(1) < 0.99 && phic(2) > -0.99
        phi = phic;
        Hphi = speye(T) - phi(1)*sparse(2:T, 1:(T-1), ones(1,T-1), T, T) ...
            - phi(2)*sparse(3:T, 1:(T-2), ones(1,T-2), T, T);
        count_phi = count_phi + 1;
    end
    
    % step 3: sample sigc2
    sigc2 = 1/gamrnd(nu_sigc2 + T/2, ...
        1/(S_sigc2 + (c-Xphi*phi)'*(c-Xphi*phi)/2));
    
    % step 4: sample sigtau2
    sigtau2 = 1/gamrnd(nu_sigtau2 + T/2, ...
        1/(S_sigtau2 + (tau-tau0)'*HH*(tau-tau0)/2));
    
    % step 5: sample tau0
    Ktau0 = 1/b0 + 1/sigtau2;
    tau0_hat = Ktau0\(a0/b0 + tau(1)/sigtau2);
    tau0 = tau0_hat + sqrt(Ktau0)'\randn;
    
    % print counter
    if (mod(isim, 1000) == 0)
        disp([num2str(isim) ' loops ']);
    end
    
    % store parameters
    if isim > burnin
        isave = isim - burnin;
        store_tau(isave, :) = tau';
        store_theta(isave, :) = [phi' sigc2 sigtau2 tau0];
    end
 
end

theta_hat = mean(store_theta);
theta_CI = quantile(store_theta, [0.025 0.975]);
tau_hat = mean(store_tau)';

%% Plots

tt = (1948:0.25:2015.75)';
dates = (datetime(1948,01,01):calquarters(1):datetime(2015,12,01))';

figure
hold on
plot(dates, y-tau_hat, 'linewidth', 1)
plot(dates, zeros(T,1), '--k', 'linewidth', 1)
recessionplot
hold off
% xlim([1948 2016])
box off
set(gcf, 'Position', [100 100 800 300])

figure
hold on
plot(dates, y, 'linewidth', 1)
plot(dates, tau_hat, 'linewidth', 1)
plot(dates, y-tau_hat, 'linewidth', 1)
recessionplot
hold off
legend('Data', 'Trend', 'Gap', 'location', 'best')
% xlim([1948 2016])
box off
set(gcf, 'Position', [100 100 800 300])




