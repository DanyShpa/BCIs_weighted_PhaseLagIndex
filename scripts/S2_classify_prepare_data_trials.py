###############################################################################
## Description: Prepare data for classification at run level for reach subject-group##                                                                ##
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
import pandas as pd
import glob
import re
import json
import fun.tools as tools
from fun.tools import find_pos_t, outlier_trials, ERP_baseline_z
import mne
from mne import io
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne_connectivity.viz import plot_sensors_connectivity, plot_connectivity_circle
import numpy as np
import pylab as plt
from random import seed
from random import choice
import winsound

#%% Clear all

os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Code/scripts')

tools.clear_all()


#%% Load matlab structure and create variables

path={'in':'../data/individual/', 
      'out':'../derivatives/classification/',
      'sub_folder':'data_training_equal/', 
      # baseline_zscore_raw; raw
      'patt':'*.mat' }

files = os.listdir(path['in'])

directory = path['out'] + path['sub_folder']


if not os.path.exists(directory):
    os.makedirs(directory)

# selec_chan=['Fz', 'Pz', 'Oz', 'Cz', 'P7', 'P3', 'P4', 'P8']


# with open(path['in']+'../Pos32Electrodes.locs') as f:
    
#     lines = f.readlines()
    
#     chanels=[[] * i for i in range(len(lines)) ]
#     s=[[] * i for i in range(len(lines)) ]
    
#     for i in range(len(lines)):
        
#         s_lines= lines[i].replace('\n','').split(' ')
        
#         chanels[i] = [int(s_lines[0]),s_lines[3],1 if s_lines[3] in selec_chan else 0]

custom_montage=mne.channels.read_custom_montage('../data/Pos32Electrodes.locs')
        
#%% Load struct matlab

# Set dict to save

classif_tot= {'id':[],
           'group':[],
           'run':[],
           'erp':[],
           'erp_type':[]}


#%% For each subject, for each run calculate P300 matrix conn and noP300 matrix conn
# Save in one matrix for all data to later classify

if os.path.exists(directory+"/outliers_recording.txt"):
    os.remove(directory+"/outliers_recording.txt")

for k,file in enumerate(files):
    
    # Load time series data
    
    data = sio.loadmat(path['in']+file)
    
    runs = np.shape(data['x'][0])[0]
    
    winsound.Beep(400, 500)
    
    for r in range(24):
        
        # Load r run from k subject
        
        x = data['x'][0][r]
                            
        y = data['y'][0][r][0]
        
        t = data['t'][0][0]   
        
        stim = data['stim'][0][r][0]
        
        
        #classif_tot['t']=t
        
        
        # Delete outlier trials
    
        f= open(directory+"/outliers_recording.txt","a")
        print('Subject_',file, file=f)
        print('Run_',r, file=f)
         
        x, y = outlier_trials(x,y,f,k )
    
        f.close()
        
        # Correct baseline
        
        x = ERP_baseline_z(x, t)
        
        
        
        #%% CALCULATE CONNECTIVITY
        
        time_range= (t>=0.0) & (t<=0.500)
        x=x[:,:,time_range[0]]
        
     
        #%% Create data structure MNE
        
        sfreq = 1024
        ch_names=custom_montage.ch_names
        ch_types = ['eeg']*len(ch_names)
        
        # Create Events structure
        n=len(x)
        events_number=np.array(range(0,n));
        events_zeros=np.zeros(n);
        
        events = np.c_[events_number,events_zeros, y].astype(int)
        
        #%% Create EpochsArray
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        tmin, tmax = 0.0, 0.5 #0.2, 0.5#t[0][0]-0.2
        
        epochs_P300 = mne.EpochsArray(x[events[:,2]==1,:,:], tmin=tmin,
        info=info, events=events[events[:,2]==1],
                                  event_id={'P300': 1}) 
        
        epochs_noP300 = mne.EpochsArray(x[events[:,2]==-1,:,:], tmin=tmin,
        info=info, events=events[events[:,2]==-1],
                                  event_id={'noP300': -1}) 
        
        #%% Compute connectivity
        
        
        con_methods  = 'wpli'
        fmin, fmax = 8., 12.
        tmin=0.0

        conn_params={'conn_methods': con_methods, 
                     'fmin': fmin,
                     'fmax': fmax,
                     'tmin': tmin,
                     'ch_names':ch_names,
                     'sfreq': sfreq}   


        # for i in range(3):
            
        #     fig, ax = plt.subplots(3,1)
            
        #     ax[0].plot(x[i,8,:]); ax[0].set_title("Chan 8")
        #     ax[1].plot(x[i,23,:]); ax[1].set_title("Chan 23")
        #     ax[2].plot(np.abs(x[i,23,:])-np.abs(x[i,8,:])); ax[2].set_title("abs(Ch23)-abs(Ch8)")
            
        #     fig.tight_layout()
        #     fig.show()
        
        # Need tmin= 100
        #RuntimeWarning: fmin=7.000 Hz corresponds to 2.803 < 5 cycles based on 
        #the epoch length 0.400 sec, need at least 0.714 sec epochs or fmin=12.488. 
        #Spectrum estimate will be unreliable.
        
        con_P300_dense = spectral_connectivity_time(
            epochs_P300, freqs=[fmin,fmax], method=con_methods, sfreq=sfreq,
            mode= 'multitaper',n_cycles=4, faverage=True)#n_cycles=3,
        
        
        con_noP300_dense = spectral_connectivity_time(
            epochs_noP300, freqs=[fmin,fmax], method=con_methods, sfreq=sfreq,
            mode= 'multitaper',n_cycles=4, faverage=True)#n_cycles=3
        
        
        
        winsound.Beep(200, 500)
        
        # Get data as connectivity matrices
        con_P300 = con_P300_dense.get_data(output="dense")
        con_noP300 = con_noP300_dense.get_data(output="dense")
        
        
        # Get flat data array for each trial and save

        for Tr, Tr_i in enumerate(con_P300):
            
            fig, ax = plt.subplots()
            
            con_P300_i= con_P300[Tr,:,:,0]
            ax.imshow(con_P300_i)
            plt.show()
            
            P300_flat = con_P300_i[np.tril_indices(len(conn_params['ch_names']), k = -1)]      
            
            # save P300 for this run and subject
            classif_tot['id'].append(file[9])
            classif_tot['group'].append( int(int(file[9])>5) ) # g_0 Disabled; g_1 Healthy
            classif_tot['run'].append(r)
            classif_tot['erp'].append(P300_flat)
            classif_tot['erp_type'].append('P300')
            
            
        for Tr,Tr_i in enumerate(con_noP300):
             
            fig, ax = plt.subplots()
            
            con_noP300_i= con_noP300[Tr,:,:,0]
                 
            noP300_flat=con_noP300_i[np.tril_indices(len(conn_params['ch_names']), k = -1)]
            ax.imshow(con_noP300_i)
            plt.show()
            
            # Save noP300 for this run and subject
            
            classif_tot['id'].append(file[9])
            classif_tot['group'].append( int(int(file[9])>5) ) # g_0 Disabled; g_1 Healthy
            classif_tot['run'].append(r)
            classif_tot['erp'].append(noP300_flat)
            classif_tot['erp_type'].append('noP300')

#%% Save data_matrix


if not os.path.exists(directory):
    os.makedirs(directory)

# In .mat format
scipy.io.savemat(directory+'data_training_indiv_trials_run.mat', classif_tot)

# In npy format
#np.savetxt(path['out']+'/conn_matrix_whole/'+'S_'+s_id, x, fmt='%1.4e')   
np.save(directory+'data_training_indiv_trials_run.npy',classif_tot)
        
        
