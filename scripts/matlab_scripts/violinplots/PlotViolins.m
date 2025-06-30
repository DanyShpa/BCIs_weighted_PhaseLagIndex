clc
clear
close all

%Method Vacular proyect
load('../../../../vascular/MeansValues.mat')
A = Data.Prec;
colorlut = [0 0 1; 0.2 0.6 1; 1 0.7 0; 0.1 1 0.5]
my_violinplot(A, 'estimator', 'Median', 'between', 0.1, 'groups_name', {'LST-lpa', 'LST-lga', 'SAMSEG', 'BIANCA'}, 'point_size', 300, 'color_scheme', colorlut);


%BD_Frontiers

load('C:\Users\berto\Desktop\Proyectos\Binge Drinking\ResultsBD_Frontiers.mat')
colorlut = [0 0 1; 1 0 0; 0 0 1; 1 0 0]
my_violinplot(A, 'estimator', 'Mean', 'between', 0.1, 'groups_name', {'CNpre', 'BDpre', 'CNpost', 'BDpost'}, 'point_size', 300, 'color_scheme', colorlut);
