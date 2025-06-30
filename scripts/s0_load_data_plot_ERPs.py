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
import scipy as sp
import math

rcParams.update({'font.size': 15})
import pandas as pd
import glob
import re
import json
import fun.tools as tools
from fun.tools import find_pos_t, outlier_trials, ERP_baseline_z

#%% Clear all

os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Code/scripts')

tools.clear_all()



#%% Load matlab structure and create variables


path={'in':'../data/individual/',
      'out':'../derivatives/time_series/individual/',
      'sub_folder':'baseline_zscore_raw', 
      # baseline_zscore_raw; raw
      'patt':'*.mat' }

files = os.listdir(path['in'])

directory = path['out'] + path['sub_folder']


if not os.path.exists(directory):
    os.makedirs(directory)

selec_chan=['Fz', 'Pz', 'Oz', 'Cz', 'P7', 'P3', 'P4', 'P8']

with open(path['in']+'../Pos32Electrodes.locs') as f:
    
    lines = f.readlines()
    
    chanels=[[] * i for i in range(len(lines)) ]
    s=[[] * i for i in range(len(lines)) ]
    
    for i in range(len(lines)):
        
        s_lines= lines[i].replace('\n','').split(' ')
        
        chanels[i] = [int(s_lines[0]),s_lines[3],1 if s_lines[3] in selec_chan else 0]
         
        
#%% Load struct matlab

# Set dict to save

subj_tot= {'id':[],
           'group':[],
           'P300':[],
           'nP300':[]}

#%% Plot Individual ERP P300 vs noP300

if os.path.exists(directory+"/outliers_recording.txt"):
    os.remove(directory+"/outliers_recording.txt")

list_P6=list()    

for k,file in enumerate(files):
    
    #file='data_subj8_sess1234.mat'
    
    # Save subject data
    # k=2
    # file='data_subj3_sess1234.mat'
    
    subj_tot['id'].append(file[9])
    
    subj_tot['group'].append( int(int(file[9])>5) ) # g_0 Disabled; g_1 Healthy
    
    # Load time series data
    
    data = sio.loadmat(path['in']+file)
    
    runs = np.shape(data['x'][0])[0]
    
    x = np.concatenate( [ data['x'][0][i] for i in range(runs) ])
                        
    y = np.concatenate([ data['y'][0][i] for i in range(runs) ],1)[0]
    
    t = data['t'][0][0]   
    
    stim = np.concatenate([ data['stim'][0][i] for i in range(runs) ],1)[0]
    
    subj_tot['t']=t
    
    # Delete outlier trials

    f= open(directory+"/outliers_recording.txt","a")
    print('Subject_',subj_tot['id'][k], file=f)
     
    x, y = outlier_trials(x,y,f,k )

    f.close()
    
    # Correct baseline
    
    x = ERP_baseline_z(x, t)
    
    
    # %% Save clean epochs for later
    subj_ind_clean= {'id':subj_tot['id'][k],
               'group':subj_tot['group'][k],
               'x':x,
               'y':y,
               'stim':stim,
               't':t,
               'ch':lines}
    
    if not os.path.exists('../derivatives/clean/'):
        os.makedirs('../derivatives/clean/')
    
    np.save('../derivatives/clean/subj_'
            +subj_tot['id'][k]+'.npy',subj_ind_clean)
    

    #%% Extract time and epochs of interest
    
    
    time_range= (t>=-0.200) & (t<=0.500)
    
    P300=y==1
    
    ch=np.array([ chanels[i][2] for i in range(len(chanels))],dtype='bool')
    
    # Raw ERP average channels
    
    subj_tot['P300'].append(
            np.mean( x[ np.ix_(P300, ch, time_range[0] )] , axis=1))
    
    subj_tot['nP300'].append(
            np.mean( x[ np.ix_(P300==False, ch, time_range[0]) ], axis=1)) 
    
    
    
    # plot time series


    for trial in range(12):
        
        fig, axs = plt.subplots(8)

        for c,chn_i in enumerate(np.where(ch)[0]):
            
            t_plot=np.linspace(-200,500,700)
            y_plot=np.transpose(x[trial,chn_i, 0:700]) 
            
            
            if P300[trial]==True:
                color='#FF0000'
            else:
                color='#003fff'
                
            axs[c].plot(t_plot,y_plot,color=color)
            axs[c].set_ylabel(chanels[chn_i][1],fontsize=12)
            
            if i>1:
                axs[c].label_outer()
                axs[c].set_yticks([])
                
        #%% Save fig
      
        plt.savefig(directory+'/trial'+str(trial)+".jpg",bbox_inches='tight', dpi=600) 
         
         # bbox_inches='tight'
         
                
    
    #%% # Dict with ERP data and types (OLD)
    
    # #* We can use axis=0 to find the mean of each column in the NumPy matrix
    
    # subj_tot['P300'].append(
    #     scipy.stats.zscore(
    #         np.mean( x[np.ix_(P300, ch, time_range[0] )], axis=1) ,axis=1))
    
    # subj_tot['nP300'].append(
    #     scipy.stats.zscore(
    #         np.mean( x[ np.ix_(P300==False, ch, time_range[0]) ], axis=1),axis=1)) 
    
    
    #%% Plot ERP average for k subject
    
    print(' Iteration File: ', k)
    
    fig, ax = plt.subplots()
    fig.tight_layout()
    #for k in range(len(files)):
        
    
    # P300
    data1=subj_tot[list(subj_tot.keys())[2]][k]
    x1 = np.arange(data1.shape[1])
    est1 = np.mean(data1, axis=0)
    se1 = (np.std(data1, axis=0)*1.96)/np.sqrt(np.shape(data1)[0])
    cis1 = (est1 - se1, est1 + se1)
    
    ax.plot(x1,est1,color='red', lw='1',
            label=list(subj_tot.keys())[2])
    ax.fill_between(x1,cis1[0],cis1[1],
                    alpha =0.1,
                    label='_Hidden label',
                    lw=0.5,
                    color='red')
    
    
    # noP300
    data=subj_tot[list(subj_tot.keys())[3]][k]
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    se = (np.std(data, axis=0)*1.96)/np.sqrt(np.shape(data)[0])
    cis = (est - se, est + se)
    
    
    ax.plot(x,est,color='blue', lw='1',
            label=list(subj_tot.keys())[3])
    ax.fill_between(x,cis[0],cis[1],
                    alpha =0.1,
                    label='_Hidden label',
                    lw=0.5,
                    color='blue')

    
    a,b = find_pos_t(x,t[time_range])
    plt.margins(x=0)
    
    plt.xticks(a, b )
    
    
    y_low,y_high=[np.min([est1,est]),np.max([est1,est])]
    
    
    
    if 'baseline' in path['sub_folder']:
        marg=2
        step= y_high
        y_lab='baseline z-scored '

            
    else:
        marg=5
        step= y_high
        y_lab='Amplitude ($\mu V$)'

            
    # y ticks
    y_low,y_high=[np.min([est1,est]),np.max([est1,est])]
    
    
    #y_tks= np.concatenate([ np.flip(np.arange(0,y_low*marg,-step),axis=0) ,
    #                      np.arange(0,y_high*marg,step)[1:]])
    y_tks= np.concatenate([ np.flip( np.arange(0,-1,-0.2),axis=0),
                            np.arange(0,1,0.2)[1:]])
    
    y_tks=np.around(y_tks, 2)
    #y_tks[

    plt.yticks(y_tks)
    
    #plt.ylim(y_low*marg,y_high*marg)
    plt.ylim(-1,1)
    
    #plt.ylabel(y_lab)
    #plt.ylabel(r"z-score")
    
    plt.xlabel(r"time (ms)")
    name='Subject '+subj_tot['id'][k]
    
    plt.grid(color='grey', linestyle='dashed',linewidth = 1, alpha = 0.25)
    plt.axhline(y=0,color='black', linestyle='dashed',linewidth = 1, alpha = 0.25)
    
    if k==0:
        plt.legend()
        
    if k==0 or k==4:
        plt.ylabel(y_lab)
        
    
    
    plt.title(name)
    
    ## save plot data
    mean_est1=list(est1)
    mean_est1.append(['P300'])
    mean_est1.append([subj_tot['id'][k] ])
    mean_est1.append([i ])
    
    list_P6.append(mean_est1)
    
    mean_est=list(est)
    mean_est.append(['no P300'])
    mean_est.append([subj_tot['id'][k] ])
    mean_est.append([i ])
    
    
    list_P6.append(mean_est)
    
    
    
    #%% Save fig
  
    plt.savefig(directory+'/'+name +".jpg",bbox_inches='tight', dpi=600)#bbox_inches='tight'
    
    plt.show()
    
#%% Save average time ERP data by subject

np.save(directory+'/subj_tot_'+path['sub_folder']+'.npy',subj_tot)


# q = np.load(directory + "subj_tot_baseline_corr_z_norm.npy",allow_pickle=True)
# q = q[()]    

db_P6=pd.DataFrame(list_P6)
db_P6.to_excel('db_P6.xlsx')






