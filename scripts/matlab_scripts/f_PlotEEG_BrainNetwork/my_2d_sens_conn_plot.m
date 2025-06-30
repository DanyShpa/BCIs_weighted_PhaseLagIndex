clear all;  clc;    close all;

% lets load several networks to be plotted
aij31 = importdata('aij31.txt');    %31 channels

%% plotting the available layouts
% figure(1);
% f_PlotEEG_BrainNetwork(31)

%% plotting brain network with nodes of equal sizes and no colorbars  
% % Here we want to threshold a network of 31 channels for fast plotting
% under different option to show the links' weights.
nch = 31; %take care of this variable (nch must be according to matrix size you want to plot it)
p = 0.03;   %proportion of weigthed links to keep for.
aij = threshold_proportional(aij31, p); %thresholding networks due to proportion p
ijw = adj2edgeL(triu(aij));             %passing from matrix form to edge list form


n_features = sum(aij, 2);
cbtype = 'wcb';

figure(1);
f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx', n_features, 'n_nn2nx', cbtype);


%close all