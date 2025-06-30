# -*- coding: utf-8 -*-
# """
# Created on Wed Aug  9 11:45:08 2023

# @author: danyl
# """

###############################################################################
## Description: Plot average connectivity by subject and group    ##
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
import pickle
import glob
import seaborn as sns

#from collections import Sequence


#%% Clear all

os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Code/scripts')

tools.clear_all()

#%% Load matlab structure and create variables


path={'in':'../derivatives/connectivity/individual/',
      'out':'../derivatives/connectivity/individual/figs/boxplots/',
      'patt':'conn*.npy' }



files = glob.glob(path['in']+path['patt'])


if not os.path.exists(path['out']):
    os.makedirs(path['out'])
    
    
conn_methods={'plv','ciplv','pli','wpli'}

data=[]

my_pal = { "P300": "#FF0000","noP300": "#0000FF"}

for m,method in enumerate(conn_methods):
    

    for s, subj in enumerate(files):
        
        
        q = np.load( files[s] ,allow_pickle=True)
        q = q[()] 
        q_keys=list(q.keys())
        
        s_id, group, con_res_P3, con_res_nP3, ch_pos, ch_names = q[q_keys[0]],q[q_keys[1]],q[q_keys[2]],q[q_keys[3]],q[q_keys[4]],q[q_keys[5]]
        
        print('Method: ', list(con_res_P3.keys())[m] )
        print('File: ',s)
        
        P300= con_res_P3[method][np.tril_indices(np.shape(con_res_P3[method])[0],-1)]
        noP300=con_res_nP3[method][np.tril_indices(np.shape(con_res_nP3[method])[0],-1)]
        
        data.append([method, group, s_id, 'P300']+P300.tolist())
        
        data.append([method, group, s_id, 'noP300']+noP300.tolist())
        
        
        
cols=['method','group','s_id','event_type']+['A' + str(i) for i in range (0,496)];    
df = pd.DataFrame(data, columns=cols)


df_long= pd.wide_to_long(df, ["A"], i=['method','group','s_id','event_type'],
                         j='conn')

df_long= df_long.reset_index();


#%% connectivity by method by group 

# controls
ax=sns.boxplot( x="method",y="A", hue="event_type",
               data=df_long.loc[df_long['group']==1,:],
               order=["plv", "ciplv","pli","wpli"],
               palette=my_pal)

ax.set_ylabel('Phase synchronization',  fontsize=14)
ax.set_xlabel('Controls',  fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Aumentar tamaño de la leyenda
ax.legend(title='Event Type', title_fontsize=15, fontsize=12)

# Save fig

directory_2save = path['out']

if not os.path.exists(directory_2save):
    os.makedirs(directory_2save)

plt.savefig(directory_2save + "connectivity_by_event_type_controls_splited.jpg",
            bbox_inches='tight', dpi=600)#bbox_inches='tight'


plt.show()        



#%% connectivity by method by group 

# disabled
ax=sns.boxplot( x="method",y="A", hue="event_type",
               data=df_long.loc[df_long['group']==0,:],
               order=["plv", "ciplv","pli","wpli"],
               palette=my_pal)


ax.set_ylabel('Phase synchronization',  fontsize=14)
ax.set_xlabel('Disabled',  fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Aumentar tamaño de la leyenda
ax.legend(title='Event Type', title_fontsize=15, fontsize=12)

# Save fig

directory_2save = path['out']

if not os.path.exists(directory_2save):
    os.makedirs(directory_2save)

plt.savefig(directory_2save + "connectivity_by_event_type_disabled_splited.jpg",
            bbox_inches='tight', dpi=600)#bbox_inches='tight'


plt.show() 


#%% individual by conn methpd

for con,method in enumerate(conn_methods):
    
    ax=sns.boxplot( x="s_id",y="A", hue="event_type",
                   data=df_long.loc[df_long['method']==method,:],
                   palette=my_pal)
    
    ax.set_ylabel('Phase synchronization',  fontsize=14)
    ax.set_xlabel('Subject',  fontsize=14)
    ax.set_title(method)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Aumentar tamaño de la leyenda
    ax.legend(title='Event Type', title_fontsize=15, fontsize=12)
    
    plt.savefig(directory_2save +"ind/conn_"+ method+ "_by_subject_splitted.jpg",
                bbox_inches='tight', dpi=600)#bbox_inches='tight'


    
    plt.show()
    
    


 
            
        
    
    
    
    
    
    
    
    
    
    
    