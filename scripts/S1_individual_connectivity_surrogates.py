# -*- coding: utf-8 -*-
# """
# Created on Wed Aug  9 11:45:08 2023

# @author: danyl
# """

###############################################################################
## Description: Load trial data and plot time series     ##
## project.                                                                  ##
## Author: danyshpa@ucm.es                           ##
## Date: 09/08/2023                                                          ##
###############################################################################

# pip install tools
# restart kernel: console mouse;  Ctrl + .

#%% Load modules

import numpy as np
import sys,os
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import glob
import re
import json
import fun.tools as tools
from fun.tools import Surrogate_conn_PhaseRandom, sigLinks
import mne
from mne import io
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity, plot_connectivity_circle
import numpy as np
import pylab as plt
from random import seed
from random import choice
import winsound


#from collections import Sequence


#%% Clear all

os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Code/scripts')

tools.clear_all()


#%% Load matlab structure and create variables


path={'in':'../derivatives/clean/',
      'out':'../derivatives/connectivity/individual/',
      'sub_folder':'baseline_zscore_raw/', 
      # baseline_zscore_raw; raw
      'patt':'*.npy' }

files = os.listdir(path['in'])

directory = path['out'] + path['sub_folder']

# if not os.path.exists(directory):
#     os.makedirs(directory)
    
custom_montage=mne.channels.read_custom_montage('../data/Pos32Electrodes.locs')
custom_montage.plot()
    
# Set dict to save
stats_conn= {'id':[],
           'group':[],
           'mean_P300':[],
           'mean_nP300':[],
           'conn_idx':[],
           't_test_stat':[],
           't_test_pval':[]}  


files=files[0:4]

for s,file in enumerate(files):
    
    print('File: ',file)
    
    q = np.load( path['in']+file ,allow_pickle=True)
    q = q[()] 
    q_keys=list(q.keys())
    
    s_id, group, x, y, stim, t, ch = q[q_keys[0]],q[q_keys[1]],q[q_keys[2]],q[q_keys[3]],q[q_keys[4]],q[q_keys[5]],q[q_keys[6]]
    
    time_range= (t>=-0.200) & (t<=0.500)
    x=x[:,:,time_range[0]]
    
    
    
    #%% Create data structure MNE
    
    sfreq = 1024
    ch_names=custom_montage.ch_names
    ch_types = ['eeg']*len(ch_names)
    
    # Create Events structure
    n=len(x)
    events_number=np.array(range(0,n));
    #print(len(events_number))
    events_zeros=np.zeros(n);
    #print(len(events_zeros))
    #print(len(epoch_type))
    
    events = np.c_[events_number,events_zeros, y].astype(int)
    #print(events[0:5,:])   
    #print('Number of events P300: ', sum(events[:,2]==1))
    
    
    #%% Select random epochs nP300
    
    # num_noP300_epochs=len(x[y==-1,:,:])
    # num_P300_epochs=len(x[y==1,:,:])
    # print(' n epochs nP300 before selection: ',num_noP300_epochs)
    # print(' n epochs P300 before selection: ',num_P300_epochs)
    
    # # choose a random element from a list
    
    # seq=events[y==-1,0]

    # selected=np.random.choice(seq,num_P300_epochs, replace=False)
    
    # seq[selected]=-1
    

    # # Select a random number of noP300 epochs to equalize to P300 epoch number
    # x_new=np.delete (x,np.where(seq>-1), axis=0) 
    # y_new=np.delete (y,np.where(seq>-1)) 
    # #stim=np.delete (stim,np.where(seq>-1)) 
    
    # print(' n epochs nP300 after selection: ',np.shape(x_new[y_new==-1,:,:])[0])
    # print(' n epochs P300 after selection: ',np.shape(x_new[y_new==1,:,:])[0])
    
    
    
    #%% Create EpochsArray
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    tmin, tmax = -0.2, 0.5#0.2, 0.5#t[0][0]
    
    epochs_P300 = mne.EpochsArray(x[events[:,2]==1,:,:], tmin=tmin,
    info=info, events=events[events[:,2]==1],
                              event_id={'P300': 1}) 
    
    epochs_noP300 = mne.EpochsArray(x[events[:,2]==-1,:,:], tmin=tmin,
    info=info, events=events[events[:,2]==-1],
                              event_id={'noP300': -1}) 
    
   

    
    #%% Compute connectivity
    
    #con_methods  = ['plv','ciplv','pli','wpli']
    con_methods  = ['pli','wpli']
    fmin, fmax = 8., 12.
    tmin=0.0

    conn_params={'conn_methods': con_methods,
                 'fmin': fmin,
                 'fmax': fmax,
                 'tmin': tmin,
                 'ch_names':ch_names,
                 'sfreq': sfreq}   

    # Need tmin= 100
    #RuntimeWarning: fmin=7.000 Hz corresponds to 2.803 < 5 cycles based on 
    #the epoch length 0.400 sec, need at least 0.714 sec epochs or fmin=12.488. 
    #Spectrum estimate will be unreliable.
    
    con_P300 = spectral_connectivity_epochs(
        epochs_P300, ch_names,method=con_methods, mode='multitaper', sfreq=sfreq,fmin=fmin,fmax=fmax,
        faverage=True,tmin=tmin, mt_adaptive=False, n_jobs=1)
    
    winsound.Beep(440, 500)
    
    con_noP300 = spectral_connectivity_epochs(
        epochs_noP300, ch_names,method=con_methods, mode='multitaper', sfreq=sfreq,fmin=fmin,fmax=fmax,
        faverage=True, tmin=tmin,mt_adaptive=False, n_jobs=1)
    
    # winsound.Beep(440, 500)
    
    # con is 3D, get the connectivity for each method
    
    con_res_P3 = dict()
    
    for method, c in zip(con_methods, con_P300):
        con_res_P3[method] = c.get_data(output='dense')[:, :, 0]
        
    con_res_nP3 = dict()
    
    for method, c in zip(con_methods, con_noP300):
        con_res_nP3[method] = c.get_data(output='dense')[:, :, 0]
    
    #%% Plot Connectivity P300 vs noP300 for s subject for 4 different conn measures
    
    # # #if s_id == 9:
    
    # vmax=1
    # vmin=0
    # n_lines=15

    
    # for m, method in enumerate(con_methods):
        
    #     print(m,method, 'Max: ', np.max(con_res_P3[method]), 'Min: ', np.min(con_res_P3[method]))

        
    #     plot_connectivity_circle(con_res_P3[method],ch_names, 
    #           n_lines=n_lines,facecolor='white', textcolor='black',
    #         node_edgecolor='black', linewidth=1.5,
    #         colormap='jet',vmin=vmin, vmax=vmax,
    #         colorbar_size=0.3,title=method+'-P300') #con_methods[0]  ,ax=ax[0]
    #     #
    #     plot_connectivity_circle(con_res_nP3[method],ch_names, 
    #           n_lines=n_lines,facecolor='white', textcolor='black',
    #         node_edgecolor='black', linewidth=1.5,
    #         colormap='jet',vmin=vmin, vmax=vmax,
    #         colorbar_size=0.3,title=method+'- noP300') #con_methods[0]  , ax=ax[1]
            
            
    #         # fname_fig = data_path+'plot_inverse_connect.png'
    #         # fig.savefig(fname_fig, facecolor=fig.get_facecolor())
    
    
    #%% Save conn matrix whole per subject
    
    # q_conn= {'id':s_id,
    #           'group':group,
    #           'con_res_P3': con_res_P3,
    #           'con_res_nP3':con_res_nP3,
    #           'ch_pos':custom_montage.get_positions()['ch_pos'],
    #           'ch_names': ch_names}
    
    # # In .mat format
    # scipy.io.savemat(path['out']+'conn_S_'+s_id+'.mat', q_conn)
    
    # # In npy format
    # #np.savetxt(path['out']+'/conn_matrix_whole/'+'S_'+s_id, x, fmt='%1.4e')   
    # np.save(path['out']+'conn_S_'+s_id+'.npy',q_conn)
    
    #%% Save stats for each subject
    
    # for m, method in enumerate(con_methods):
        
    #     stats_conn['id'].append(s_id)
    #     stats_conn['group'].append(group)
        
    #     v1= con_res_P3[method][np.tril_indices(np.shape(con_res_P3[method])[0],-1)]
    #     v2= con_res_nP3[method][np.tril_indices(np.shape(con_res_nP3[method])[0],-1)]
        
    #     stats_conn['mean_P300'].append(np.mean(v1))
    #     stats_conn['mean_nP300'].append(np.mean(v2))
    #     stats_conn['conn_idx'].append(method) 
        
    #     scipy.stats.ttest_ind(v1,v2)
    #     stats_conn['t_test_stat'].append(scipy.stats.ttest_ind(v1,v2)[0])
    #     stats_conn['t_test_pval'].append(scipy.stats.ttest_ind(v1,v2)[1])
        
        
    #     print('S_',s_id,'; Method:',method)
        
    # *** Conclusion: Use 'wpli'  
    
    #%% SURROGATES: Save conn matrix of phase randomized epochs
    n_rep=50
    c_idx='wpli'
    con_res_P3 = con_res_P3[c_idx]
    con_res_nP3 = con_res_nP3[c_idx]
    
    conn_params['conn_methods']='wpli'
    
    # P300
    surr_P300, rand_P3 = Surrogate_conn_PhaseRandom(x[events[:,2]==1,:,:], n_rep, conn_params)
    
    # # np P300
    surr_noP300, rand_nP3 = Surrogate_conn_PhaseRandom(x[events[:,2]==-1,:,:], n_rep, conn_params)
    
    # # #%% Sig links: compare each link with n_rep surrogate links
    
    # Flatten original conn
    con_flat_P3=con_res_P3[np.tril_indices(len(conn_params['ch_names']), k = -1)]
    con_flat_nP3=con_res_nP3[np.tril_indices(len(conn_params['ch_names']), k = -1)]
    
   
    
    
    #%% Sig Links
    
    thres= 95.00
    
    # P300
    sig_P300, sig_mat_P300 = sigLinks(con_flat_P3,surr_P300,thres) #np.shape(np.where(sig_P300)) 
    np.shape(np.nonzero(sig_P300)) 
    
    # no P300
    sig_noP300, sig_mat_noP300 = sigLinks(con_flat_nP3,surr_noP300, thres)
    np.shape(np.where(sig_noP300))
    
    # Diff P300-nP300
    
    sig_diff_P300, sig_mat_diff_P300 = sigLinks(abs(con_flat_P3-con_flat_nP3),
                             abs(surr_P300-surr_noP300), thres)
    
    np.shape(np.nonzero(sig_diff_P300))

    #%% Save info for each subject
    
    out_conn= {'id':s_id,
               'group':group,
               'con_flat_P3':con_flat_P3,
               'con_flat_nP3':con_flat_nP3,
               'surr_P300':surr_P300,
               'surr_noP300':surr_noP300,
               'sig_mat_P300':sig_mat_P300,
               'sig_mat_noP300':sig_mat_noP300, 
               'sig_mat_diff_P300':sig_mat_diff_P300} 
    
    dir_surr=path['out']+'sig_links_surrogate/'
    
    if not os.path.exists(dir_surr):
        os.makedirs(dir_surr)
    
    # In .mat format
    scipy.io.savemat(dir_surr+'conn_surr_S_'+s_id+'.mat', out_conn)
    
    # In npy format
    #np.savetxt(path['out']+'/conn_matrix_whole/'+'S_'+s_id, x, fmt='%1.4e')   
    np.save(dir_surr+'conn_surr_S_'+s_id+'.npy',out_conn)
    
    

#%% Stats  ALL SUBJECTS   (out of loop)

# df=pd.DataFrame.from_dict(stats_conn  ) 
 
# df['pval_sig']=df['t_test_pval']<0.05
 
# df1=df.groupby(['conn_idx','group'])['t_test_pval', 't_test_stat'].mean()
# print(df1)

# # Cuantos sujetos hay diff sig entre conds P3 por cada grupo por cada index
# df2=df.groupby(['conn_idx','group'])['pval_sig'].sum()
# df1['num_subj_sig']=df2
# print(df2)  

# df3=df.loc[df['conn_idx']=='wpli',:]
# print(df3)  

# with pd.ExcelWriter('../derivatives/results/which_conn_idx.xlsx') as excel_writer:
#     df1.to_excel(excel_writer, sheet_name='Sheet1', index=True)
#     df2.to_excel(excel_writer, sheet_name='Sheet2', index=True)
#     df3.to_excel(excel_writer, sheet_name='Sheet3', index=True)





#print hubs
    
for i, hub in enumerate (hubs['hub_by_cond']):
    print(hubs['hub_by_cond'][str(hub)])
    print(hub)
    
    
