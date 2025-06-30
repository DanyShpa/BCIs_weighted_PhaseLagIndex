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
import itertools
from collections import Counter
#%% Clear all

#os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Python_code/scripts')

tools.clear_all()

custom_montage=mne.channels.read_custom_montage('../data/Pos32Electrodes.locs')


#%% Set path

path={'in':'../derivatives/connectivity/group/stats/',
      'out':'../derivatives/classification/data_training/',
      # baseline_zscore_raw; raw,recursive=True
      'patt':'*.npy' }

data= np.load( path['in']+'stats_all.npy', allow_pickle=True )
data=data[()]


hubs= {'hub_by_subject':[],
            'hub_P300_noP300':[], 
            'hub_by_cond':[]}


#%% Plot hubs (chanels with higher centrality)

# Fornito, A., Zalesky, A., & Bullmore, E. T. (2016). Centrality and hubs. 
# Fundamentals of brain network analysis, 137-161.


# Li, F., Tao, Q., Peng, W., Zhang, T., Si, Y., Zhang, Y., ... & Xu, P. (2020). 
# Inter-subject P300 variability relates to the efficiency of brain networks 
# reconfigured from resting-to task-state: evidence from a simultaneous event-related 
# EEG-fMRI study. NeuroImage, 205, 116285.

#%% Distribution most common hubs P300 v2 no P300 for all subjects 

# betweenness_cen_ch
# betweenness_cen_mean
# closeness_cen_ch
# closeness_cen_mean
# degree_cen_ch
# degree_cen_mean
# erp
# group
# id
# thres

logi_erp=  np.array([True if('nP300' in ele) else False for ele in data['erp'] ])

filter_hub_P300=  (np.array(data['thres'])==95) & logi_erp

filter_hub_nP300=  (np.array(data['thres'])==95) & (~logi_erp)

#%% for each subject



hubs_P3= [ data['degree_cen_ch'][pos]  
          for pos,log in enumerate(filter_hub_P300) if log]


hubs_nP3= [ data['degree_cen_ch'][pos]  
          for pos,log in enumerate(filter_hub_nP300) if log]


hub_by_subject=dict()

for i in range(len(hubs_P3)):
    hub_by_subject[str(i)]= np.unique(hubs_nP3[i] + hubs_P3[i])
    
hubs['hub_by_subject']= hub_by_subject

#%% hubs by cond

D=[]
C=[]

for i in range(len(hubs_P3)):
    
    if i < 4 :
        print(i)
        D.append(np.unique(hubs_nP3[i] + hubs_P3[i]))
    else:
        C.append(np.unique(hubs_nP3[i] + hubs_P3[i]))
        
flat_D_count=Counter(list(itertools.chain(*D)))
flat_C_count=Counter(list(itertools.chain(*C)))

flat_D_most_common_2= Counter(dict(filter(lambda x: x[1] > 1, 
                                        flat_D_count.items())))

flat_C_most_common_2= Counter(dict(filter(lambda x: x[1] > 1, 
                                        flat_C_count.items())))

hub_by_cond={'Disabled':list(flat_D_most_common_2.keys()),
              'Control':list(flat_C_most_common_2.keys())}


hubs['hub_by_cond']= hub_by_cond

#%% All data  HUBS P300 vs NO P300

flat_hubs_P3 = list(itertools.chain(*hubs_P3))
flat_hubs_nP3 = list(itertools.chain(*hubs_nP3))


hub_counts_nP3 = Counter(flat_hubs_nP3)
df_nP3 = pd.DataFrame.from_dict(hub_counts_nP3, orient='index')
df_nP3=df_nP3.sort_values(0, ascending=False)
df_nP3.plot(kind='bar',legend=False, title='most common noP300 Hubs')
plt.savefig(path['out']+'hubs/'+"hubs_nP3_degree_cen.jpg",bbox_inches='tight', dpi=200)#bbox_inches='tight'


hub_counts_P3 = Counter(flat_hubs_P3)
df_P3 = pd.DataFrame.from_dict(hub_counts_P3, orient='index')
df_P3=df_P3.sort_values(0, ascending=False)
df_P3.plot(kind='bar', legend=False, title='most common P300 Hubs')
plt.savefig(path['out']+'hubs/'+"hubs_P3_degree_cen.jpg",bbox_inches='tight', dpi=200)#bbox_inches='tight'



hub_counts_nP3_most= Counter(dict(filter(lambda x: x[1] > 1, 
                                        hub_counts_nP3.items())))

hub_counts_P3_most= Counter(dict(filter(lambda x: x[1] > 1, 
                                        hub_counts_P3.items())))




hub_by_erp={'P300':list(hub_counts_P3_most.keys()),
              'noP300':list(hub_counts_nP3_most.keys()),
              'all_unique':np.unique(list(hub_counts_P3_most.keys())+list(hub_counts_P3_most.keys()))}

hubs['hub_by_erp']= hub_by_erp


#%% Save hubs
directory=path['out']+'hubs/'

if not os.path.exists(directory):
    os.makedirs(directory)

# In .mat format
scipy.io.savemat(directory+'hubs_degree_cen.mat', hubs)

# In npy format  
np.save(directory+'hubs_degree_cen.npy',hubs)









