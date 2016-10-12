%Example of a sinusoid-like GP estimation with gaussian likelihood
clear all, close all

% Generate the data
n = 200;
displac = 5;
noise = 0.5;

x = -20:0.2:20; 
y = displac + normrnd(0,noise,1,length(x));

figure(1)
plot(x, y);

% Estimate parameters and interpolate points
m_est = mean(y);
meanfunc = {@meanConst}; hyp.mean = [m_est];
 
covfunc = {@covSum, {@covConst, @covSEiso}}; hyp.cov = log([displac, 1, 1]);
likfunc = @likGauss; sn = noise; hyp.lik = log(sn);
      
% Separate training data and test to estimate
A = randperm(length(x));
split = mat2cell(A,1,[floor(length(x)/3) length(x) - floor(length(x)/3)]);

xtr = x(split{2})'; ytr = y(split{2})';
xts = x(split{1})'; yts = y(split{1})';

% Likelihood with exact parameters
nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xtr, ytr);

nlml

figure(2)
set(gca, 'FontSize', 12)
plot(xtr, ytr, '+', 'MarkerSize', 12)
axis([-21 21 -5 15])
grid on

pred_x = sort(xts);
[mta, sts] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xtr, ytr, pred_x);

figure(3)
f = [mta+2*sqrt(sts); flipdim(mta-2*sqrt(sts),1)];
fill([pred_x; flipdim(pred_x,1)], f, [7 7 7]/8);
hold on; 
plot(pred_x, mta, 'LineWidth', 1); 
plot(xts, yts, '+', 'MarkerSize', 12)

% Optimize parameters

hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xtr, ytr);
nlml_opt = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xtr, ytr);

nlml_opt

[mta, sts] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xtr, ytr, pred_x);

figure(4)
f = [mta+2*sqrt(sts); flipdim(mta-2*sqrt(sts),1)];
fill([pred_x; flipdim(pred_x,1)], f, [7 7 7]/8);
hold on; 
plot(pred_x, mta, 'LineWidth', 1); 
plot(xts, yts, '+', 'MarkerSize', 12)

% Making the optimization a bit harder
m_est = mean(y);
hyp.mean = [m_est + normrnd(0,1)];
hyp.cov = log([displac + normrnd(0,1), 1, 1]);
hyp.lik = log(sn + 1);

hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, xtr, ytr);
nlml_opt2 = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xtr, ytr);

nlml_opt2

[mta, sts] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, xtr, ytr, pred_x);

figure(5)
f = [mta+2*sqrt(sts); flipdim(mta-2*sqrt(sts),1)];
fill([pred_x; flipdim(pred_x,1)], f, [7 7 7]/8);
hold on; 
plot(pred_x, mta, 'LineWidth', 1); 
plot(xts, yts, '+', 'MarkerSize', 12)