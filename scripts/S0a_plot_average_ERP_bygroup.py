# -*- coding: utf-8 -*-
# """
# Created on Tue Aug 15 18:49:13 2023

# @author: danyl
# """
###############################################################################
## Description: Plot average ERP P300 and noP300 by group healthy and disabled     ##
## Author: danyshpa@ucm.es                           ##
## Date: 09/08/2023                                                          ##
###############################################################################

# restart kernel: console mouse;  Ctrl + .
#%% Load modules
import numpy as np
import sys,os
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 18})
import pandas as pd
import fun.tools as tools
from fun.tools import find_pos_t
import glob
import re
import json
from itertools import compress

#%% Clear all

#os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Python_code/scripts')

tools.clear_all()

#%% Set path

path={'in':'../data/individual/',
      'out':'../derivatives/time_series/individual/',
      # baseline_zscore_raw; raw
      'patt':'*.npy' }

files = glob.glob(path['out']+'**/'+path['patt'],recursive=True)

#%% Load average ERP for each subject subjects

#file=files[1]
labels=['Disabled','Control']
erp=['P300','nP300']

#list_P6=list()

for i, file in enumerate(files):
    
    print('File: ',file)
    
    q = np.load( file ,allow_pickle=True)
    q = q[()] 

    #%% Plot averaged by group
    fig, ax = plt.subplots(1,2,figsize=(20,6))
    #fig.tight_layout()
    fig.subplots_adjust(top=0.9)  
    
    for j,lab in enumerate(labels):
        print('Label: ',lab)
        
        for k, erp_k in enumerate(erp):
            print('erp: ',erp_k)
    
            # Get avg k ERP for j group 
            
            y = np.array(q['group'] )
    
            p3 = np.concatenate([q[erp_k][a] for a in np.where(y==j)[0]] )

            # Time
            t = q['t']
            time_range = (t>=-0.201) & (t<=0.501)
            
            # Plot average ERP P300 for each group
            
            data=p3
            
            x = np.arange(data.shape[1])
            est = np.mean(data, axis=0)
            sd = (np.std(data, axis=0)*1.96)/np.sqrt(np.shape(data)[0])
            cis = (est - sd, est + sd)
            
            
            color='red' if k==0 else 'blue' 
            
            ax[j].plot(x,est,color=color, lw='1',
                    label=erp_k)
            
            ax[j].fill_between(x,cis[0],cis[1],
                            alpha =0.1,
                            label='_Hidden label',
                            lw=0.5,
                            color=color)
            
            # Axis plot
            a,b=find_pos_t(x,t[time_range])
            ax[j].margins(x=0)
            ax[j].set_xticks(a, b)
            
            # y ticks
            y_low,y_high=[np.min(est),np.max(est)]
            
            marg=3
        
            if 'baseline' in file:
                marg=6
                if k==0:
                    
                    marg=4
                    
                    tit= 'Baseline z-score averaged  ERP'
                    step=y_high
                    y_lab='baseline z-scored '
                    y_tks= np.concatenate([ np.flip(np.arange(0,y_low*marg,-step),axis=0) ,
                                  np.arange(0,y_high*marg,step)[1:]])
                    y_tks=np.around(y_tks, 2)
                    
                    
                    
            else:
                if k==0:
                    tit= 'Averaged ERP'
                    step=y_high
                    y_lab='Amplitude ($\mu V$)'
                y_tks= np.concatenate([ np.flip(np.arange(0,y_low*marg,-step),axis=0) ,
                                  np.arange(0,y_high*marg,step)[1:]])
                y_tks=np.around(y_tks, 2)
                
            #y_tks[
        
            ax[j].set_yticks(y_tks)
            ax[j].set_yticklabels(y_tks)
            
            ax[j].set_ylim(y_low*marg,y_high*marg)
            
            
            
            ax[j].set_xlabel(r"time (ms)")

            #ax[j].set_ylabel(r"z-score")
            
            ax[j].grid(color='grey', linestyle='dashed',linewidth = 1, alpha = 0.25)
            ax[j].axhline(y=0,color='black', linestyle='dashed',linewidth = 1, alpha = 0.25)
            
            if j==1 :
                ax[j].legend()
                ax[j].set_ylabel(y_lab)
                
            
            ax[j].set_title(lab)
            
            #tit=file.split('/')[-1].split('\\')[-1].replace('_',' ')[:-4]
                                      
            fig.suptitle(tit)
            
            
            
            
            
            
            
            #%% Save fig
            
            directory_2save = path['out'].replace('individual','group')
            
            if not os.path.exists(directory_2save):
                os.makedirs(directory_2save)
          
            plt.savefig(directory_2save + file.split('\\')[-1] .split('.')[-2]+".jpg",
                        bbox_inches='tight', dpi=150)#bbox_inches='tight'
            
            #plt.show()
            

            
            
            
        
           
            
        
        
        
        
    
    
    
    
    
    
    
    # # Get avg g_1 Normal
        
        # p3_g1 = np.mean(np.concatenate([q['P300'][j] for j in np.where(y==0)[0]]),0)
        
        # pn3_g1 = np.mean(np.concatenate([q['nP300'][j] for j in np.where(y==0)[0]]),0)



