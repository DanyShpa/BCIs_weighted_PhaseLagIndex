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

#%% Load average ERP for each subject subjects

#file=files[1]
labels=['Disabled','Control']
erp=['P300','nP300']#, 
con_flat_dict=['con_flat_P3','con_flat_nP3']
con_surr_dict=['surr_P300','surr_noP300']


subj_tot={}

sig_conn_Groups= {'group':[],
           'erp':[],
           'pval_sig':[],
           'sigLks':[],
           'sig_mat':[],
           'numel_sigLks':[],
           'mean_conn_sigLks':[],
           'netwk_centrality':[]} 


thres= [99.99, 99, 97.5, 95, 92.5]

thres_i=thres[4]

#Change manual IMPORTANTE 
numel_surr= sample(range(200), 50)

#%% Get sigLinks matrix averaged by group
 
    
for j,lab in enumerate(labels):
    print('Label: ',lab)
    
    for k, erp_k in enumerate(erp):
        print('erp: ',erp_k)
        
        con_flat_all=[]
        surr_all=[]

        # Get avg k conn for j group 
        for i, file in enumerate(files):

            
            q = np.load( path['in']+files[i] ,allow_pickle=True)
            q = q[()] 
            
            # Extract k data from j group
            
            if q['group']==j:
                
                print('File: ',file)
            
                con_flat_all.append(q[con_flat_dict[k]])
                
                surr_all.append(q[con_surr_dict[k]])

            
        con_flat_all= np.mean(np.concatenate([con_flat_all[a].reshape(1,-1) 
                                              for a in range(len(con_flat_all))]),0)
        
        surr_all=np.concatenate([surr_all[a] for a in range(len(surr_all))])
        
        
        if k == 0: 
            con_flat_P3= con_flat_all
            surr_P300 = surr_all
            
        #%% Sig Links
        
        # ERP type
        sigLks, sig_mat = sigLinks(con_flat_all,surr_all,thres_i, numel_surr) #np.shape(np.where(sig_P300)) 
        print('Number links cond '+erp_k+': '+ str( np.shape(np.nonzero(sigLks))[1] ))
        print('mean links cond '+erp_k+': '+ str( np.mean(con_flat_all[np.nonzero(sigLks)]))[:6] )
              
        sig_conn_Groups['group'].append(lab)
        sig_conn_Groups['erp'].append(erp_k)
        sig_conn_Groups['sigLks'].append( sigLks)
        sig_conn_Groups['pval_sig'].append(thres_i)
        sig_conn_Groups['sig_mat'].append( sig_mat)
        sig_conn_Groups['numel_sigLks'].append( np.shape(np.nonzero(sigLks))[1])
        sig_conn_Groups['mean_conn_sigLks'].append( np.mean(con_flat_all[np.nonzero(sigLks)]))
        sig_conn_Groups['netwk_centrality'].append( network_centrality(sig_mat, custom_montage.ch_names))
        
        
        if k==1: # Diff P300-nP300
        
            diff_abs=abs(con_flat_P3-con_flat_all)
            diff_surr_abs=abs(surr_P300-surr_all)
            
            sig_diff, sig_mat_diff = sigLinks(diff_abs,
                                      diff_surr_abs, thres_i, numel_surr)
            
            print('Number links cond diff '+'diff (P300-nP300)'+': '+ str( np.shape(np.nonzero(sig_diff))[1] ))
            print('mean links cond diff'+'diff (P300-nP300)'+': '+ str( np.mean(diff_abs[np.nonzero(sig_diff)]))[:6] )
            

            sig_conn_Groups['group'].append(lab)
            sig_conn_Groups['erp'].append('diff (P300-nP300)')
            sig_conn_Groups['sigLks'].append( sig_diff)
            sig_conn_Groups['pval_sig'].append(thres_i)
            sig_conn_Groups['sig_mat'].append( sig_mat_diff)
            sig_conn_Groups['numel_sigLks'].append( np.shape(np.nonzero(sig_diff))[1])
            sig_conn_Groups['mean_conn_sigLks'].append( np.mean(diff_abs[np.nonzero(sig_diff)]))
            sig_conn_Groups['netwk_centrality'].append( network_centrality(sig_mat_diff, custom_montage.ch_names))
            
            
            
#%% Save sig links matrix

            
dir_group_sigLks=path['out']+'../group/'

if not os.path.exists(dir_group_sigLks):
    os.makedirs(dir_group_sigLks)

# In .mat format
scipy.io.savemat(dir_group_sigLks+'sigLks_'+str(thres_i).replace('.','_')+'.mat', 
                 sig_conn_Groups)

# In npy format
#np.savetxt(path['out']+'/conn_matrix_whole/'+'S_'+s_id, x, fmt='%1.4e')   
np.save(dir_group_sigLks+'sigLks_'+str(thres_i).replace('.','_')+'.npy',sig_conn_Groups)



 
 

            
