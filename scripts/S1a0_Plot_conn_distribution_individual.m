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


#files=files[0:4]

for s,file in enumerate(files):
    
    print('File: ',file)