# -*- coding: utf-8 -*-
# """
# Created on Wed Aug  9 11:45:08 2023

# @author: danyl
# """

#%% Load modules
import numpy as np
import sys,os
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import fun.tools as tools
from fun.tools import find_pos_t, sigLinks, network_centrality
from random import choice, sample
import glob
import mne
import re
import json
from itertools import compress
import networkx as nx
#%% Clear all

#os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Python_code/scripts')

tools.clear_all()

custom_montage=mne.channels.read_custom_montage('../data/Pos32Electrodes.locs')


#%% Set path

path={'in':'../derivatives/connectivity/individual/sig_links_surrogate/',
      'out':'../derivatives/connectivity/individual/',
      # baseline_zscore_raw; raw,recursive=True
      'patt':'*.npy' }

files = [os.path.basename(i) for i in glob.glob(path['in'] + path['patt'])]

#%% Set params

#file=files[1]
labels=['Disabled','Control']
erp=['P300','nP300']#, 
con_flat_dict=['con_flat_P3','con_flat_nP3']
con_surr_dict=['surr_P300','surr_noP300']


thres= [99.99, 99, 97.5, 95, 92.5]

#Change manual IMPORTANTE 
numel_surr= sample(range(50), 50)

#%% Plot  individual and group differences
                     

stats_conn= {'id':[],
            'group':[],
            'erp':[], 
            'thres':[],
            'mean_conn':[],
            'sd_conn':[],
            'n_conn_sigLks':[],
            'degree_cen_mean':[],
            'closeness_cen_mean':[],
            'betweenness_cen_mean':[],
            'degree_cen_ch':[],
            'closeness_cen_ch':[],
            'betweenness_cen_ch':[]} 
            
            
for i, file in enumerate(files):
    print(' \n \nSubject: ',file,'\n \n')
    
    for k, erp_k in enumerate(erp):
        print('\n \n erp: ',erp_k)
        
        con_flat_all=[]
        surr_all=[]

        #  calculate measures stats for each of the thresholds

        q = np.load( path['in']+files[i] ,allow_pickle=True)
        q = q[()] 
        
        con_flat= q[con_flat_dict[k]]

        surr_flat= q[con_surr_dict[k]]
        
        # Get sigLks for t threshold and
        
        for t,thres_i in enumerate(thres ):
            print('thres: ',thres_i)
            
            if k == 0: 
                con_flat_P3= con_flat
                surr_flat_P300 = surr_flat
            
            # ERP type
            sigLks, sig_mat = sigLinks(con_flat,surr_flat,thres_i, numel_surr) #np.shape(np.where(sig_P300)) 
            print('Number links cond '+erp_k+': '+ str( np.shape(np.nonzero(sigLks))[1] ))
            #print('mean links cond '+erp_k+': '+ str( np.mean(con_flat[np.nonzero(sigLks)]))[:6] )
               
            ntwk_cen=network_centrality(sig_mat, custom_montage.ch_names)
            
            list(con_flat[np.nonzero(sigLks)])
            
            stats_conn['id'].append(q['id'])
            stats_conn['group'].append(q['group'])
            stats_conn['erp'].append(erp_k) 
            stats_conn['thres'].append(thres_i)
            stats_conn['mean_conn'].append(np.mean(con_flat[np.nonzero(sigLks)]))
            stats_conn['sd_conn'].append(np.std(con_flat[np.nonzero(sigLks)]))
            stats_conn['n_conn_sigLks'].append(np.shape(np.nonzero(sigLks))[1])
            stats_conn['degree_cen_mean'].append(np.mean(list(ntwk_cen['degree'].values())[0:5]))
            stats_conn['closeness_cen_mean'].append(np.mean(list(ntwk_cen['closeness'].values())[0:5]))
            stats_conn['betweenness_cen_mean'].append(np.mean(list(ntwk_cen['betweenness'].values())[0:5]))
            stats_conn['degree_cen_ch'].append(list(ntwk_cen['degree'].keys())[0:5])
            stats_conn['closeness_cen_ch'].append(list(ntwk_cen['closeness'].keys())[0:5])
            stats_conn['betweenness_cen_ch'].append(list(ntwk_cen['betweenness'].keys())[0:5])
            
                
                # if k==1: # Diff P300-nP300
                
                #     diff_abs=abs(con_flat_P3-con_flat)
                #     diff_surr_abs=abs(surr_flat_P300-surr_flat)
                    
                #     sig_diff, sig_mat_diff = sigLinks(diff_abs,
                #                               diff_surr_abs, thres_i)
                    
                #     #print('Number links cond diff '+'diff (P300-nP300)'+': '+ str( np.shape(np.nonzero(sig_diff))[1] ))
                #     #print('mean links cond diff'+'diff (P300-nP300)'+': '+ str( np.mean(diff_abs[np.nonzero(sig_diff)]))[:6] )
                    
                #     ntwk_cen=network_centrality(sig_mat_diff, custom_montage.ch_names)
                    
                #     stats_conn['id'].append(q['id'])
                #     stats_conn['group'].append(q['group'])
                #     stats_conn['erp'].append('sig_mat_diff_P300') 
                #     stats_conn['thres'].append(thres_i)
                #     stats_conn['mean_conn'].append(np.mean(diff_abs[np.nonzero(sig_diff)]))
                #     stats_conn['n_conn_sigLks'].append(np.shape(np.nonzero(sig_diff))[1])
                #     stats_conn['degree_cen_mean'].append(np.mean(list(ntwk_cen['degree'].values())[0:5]))
                #     stats_conn['closeness_cen_mean'].append(np.mean(list(ntwk_cen['closeness'].values())[0:5]))
                #     stats_conn['betweenness_cen_mean'].append(np.mean(list(ntwk_cen['betweenness'].values())[0:5]))
                #     stats_conn['degree_cen_ch'].append(list(ntwk_cen['degree'].keys())[0:5])
                #     stats_conn['closeness_cen_ch'].append(list(ntwk_cen['closeness'].keys())[0:5])
                #     stats_conn['betweenness_cen_ch'].append(list(ntwk_cen['betweenness'].keys())[0:5])
                    
                
                
                
#%% Save sig links matrix

dir_group_sigLks=path['out']+'../group/stats/'

if not os.path.exists(dir_group_sigLks):
    os.makedirs(dir_group_sigLks)

# In .mat format
scipy.io.savemat(dir_group_sigLks+'stats_all.mat', stats_conn)

# In npy format  
np.save(dir_group_sigLks+'stats_all.npy',stats_conn)


#%% Plot hubs (chanels with higher centrality)

# Fornito, A., Zalesky, A., & Bullmore, E. T. (2016). Centrality and hubs. 
# Fundamentals of brain network analysis, 137-161.


# Li, F., Tao, Q., Peng, W., Zhang, T., Si, Y., Zhang, Y., ... & Xu, P. (2020). 
# Inter-subject P300 variability relates to the efficiency of brain networks 
# reconfigured from resting-to task-state: evidence from a simultaneous event-related 
# EEG-fMRI study. NeuroImage, 205, 116285.

## Distribution most common hubs P300 v2 no P300





















