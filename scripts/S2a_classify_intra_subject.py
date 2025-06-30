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
import sklearn

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate,KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection import ValidationCurveDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler


#%% Clear all

#os.chdir('C:/Users/danyl/OneDrive/Documents/DANYLYNA/UNIVERSIDAD/MASTER_INT_COMPUTACIONAL/TFM2/Python_code/scripts')

tools.clear_all()

custom_montage=mne.channels.read_custom_montage('../data/Pos32Electrodes.locs')


#%% Set path

path={'in':'../derivatives/classification/data_training_equal/',
      'out':'../derivatives/classification/data_test/',
      #'sub_folder':'intra_subject',
      'patt':'*.npy' }

#%% Load data training test

data = np.load(path['in']+'data_training_indiv_trials_run.npy',allow_pickle=True)
data=data[()]

# np shape X = 368 conn matrices flat (P300/noP300) x 496 links
# 46 examples / subject: 23 runs P300 23 runs nP300
X= np.array(data['erp'] )

# erp type (P300 no P300) shape (368,)
Y= np.array(['P300' == data['erp_type'][i] for i in range( len(data['erp_type']))])
Y= Y.astype(int)



#%% Load features (NOT USED)

#hubs_file='hubs_degree_cen.npy' # hubs_betweenness_cen.npy

#hubs = np.load(path['in']+'hubs/'+ hubs_file ,allow_pickle=True)

#hubs=hubs[()]


#%% Set links names (NOT USED)

conn_names=np.empty((32,32))

ch=custom_montage.ch_names

ch_names= np.array([np.char.add(np.char.add(ch,'-')[i],ch) 
                         for i in range(len(ch))])


#%% Extract Features GROUP CONNECTIVITY

pth='../derivatives/connectivity/group/sigLks_surrogate_thres_i/'

sig92_5=np.load(pth+'sigLks_92_5.npy', allow_pickle=True)
sig92_5=sig92_5[()]

sig95=np.load(pth+'sigLks_95.npy', allow_pickle=True)
sig95=sig95[()]

sig97_5=np.load(pth+'sigLks_97_5.npy', allow_pickle=True)
sig97_5=sig97_5[()]

sig99=np.load(pth+'sigLks_99.npy', allow_pickle=True)
sig99=sig99[()]

sig99_99=np.load(pth+'sigLks_99_99.npy', allow_pickle=True)
sig99_99=sig99_99[()]


#%% Classify for each subject; 46 examples / subject: 23 runs P300 23 runs nP300

keys_filtering=['id', 'group', 'run']
df_filter = pd.DataFrame.from_dict({key: data[key] for key in keys_filtering })

 
#%% CONTROL group

#%% Classify each subject with its CONTROL group features P300 noP300

n= 4 # subj

rep= 10

features_P3= np.array(np.nonzero(sig95['sigLks'][3]))[0]
features_nP3= np.array(np.nonzero(sig95['sigLks'][3]))[0]

feat_cat=np.concatenate((features_P3,features_P3),0)

sc={'s_i':[],
    'accuracy':[],
    'f1':[],
    'roc_auc':[]}

sc_tot={'s_i':[],
    'accuracy':[],
    'accuracy_sd':[],
    'f1':[],
    'f1_sd':[],
    'roc_auc':[],
    'roc_auc_sd':[]}


# #%% send FBR data

s_ID = [eval(i) for i in data['id']]

X_df=pd.DataFrame( X[:,features_P3], columns=features_P3 )

X_df['Y']=Y
X_df['subject_ID']=s_ID

np.save('X_df.py',X_df)


#%%


for i, s_i in enumerate( np.unique(df_filter['id'])[n:] ): # Select all subjects from n to :
    
    Xi= X[ np.ix_(df_filter['id'] == s_i, features_P3 ) ]   #range(1,496) extract X data for training from subject s_i and features P300 network
    
    Yi= Y[ df_filter['id'] == s_i ]  
    
    # Create a RandomUnderSampler object
    rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
    Xi, Yi = rus.fit_resample(Xi, Yi)
    
    for j in range(rep):
        
        # ValidationCurveDisplay.from_estimator(svm.SVC(kernel="linear"), 
        #                                       Xi, Yi, param_name="C", 
        #                                       param_range=np.logspace(-7, 3, 10))
        
        #rng = np.random.RandomState(0)
        
        clf = svm.SVC(kernel='linear', C=1)#0.5)
        #clf= LinearDiscriminantAnalysis()
        scores = cross_validate(clf, Xi, Yi, cv=StratifiedKFold(10,shuffle=True), 
                                scoring=['accuracy','f1','roc_auc'])
        
        sc['s_i'].append(s_i)
        sc['accuracy'].append(np.round(np.mean(scores['test_accuracy']), 4))
        sc['f1'].append(np.round(np.mean(scores['test_f1']), 4))
        sc['roc_auc'].append(np.round(np.mean(scores['test_roc_auc']), 4))
                     
        print('Subject',s_i,'acc: ',np.mean(scores['test_accuracy']),
                      'f1', np.mean(scores['test_f1']),
                      'roc_auc',np.mean(scores['test_roc_auc']))
        
        
    sc_tot['s_i'].append(s_i)
    sc_tot['accuracy'].append( np.round( np.mean(sc['accuracy']), 4 ))
    sc_tot['accuracy_sd'].append( np.round( np.std(sc['accuracy']), 4 ))
    sc_tot['f1'].append(np.round( np.mean(sc['f1']), 4))
    sc_tot['f1_sd'].append(np.round( np.std(sc['f1']), 4))
    sc_tot['roc_auc'].append(np.round(np.nanmean(sc['roc_auc']), 4))
    sc_tot['roc_auc_sd'].append(np.round(np.nanstd(sc['roc_auc']), 4))
    

## plot PCA

#  PCA all

Xi_sig_Links=X[ : , features_P3  ]
pca = PCA(n_components=3)
pca.fit( np.transpose(Xi_sig_Links) )

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
PCA_sig_links=pca.components_

#  PCA P300
pca = PCA(n_components=3)
pca.fit(Xi[Yi==1])

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
P300_PCA_links=pca.components_

#  PCA noP300
pca = PCA(n_components=3)
pca.fit(Xi[Yi==0])

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
noP300_PCA_links=pca.components_


# PLOT TRI

from mpl_toolkits.mplot3d import Axes3D



fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect(1)
#ax = fig.add_subplot(111, projection='2d')

n = 100

xs = PCA_sig_links[0]
ys = PCA_sig_links[1]
zs = PCA_sig_links[2]

Yi_color = [ 'r' if i>0 else 'b' for i in Y ]
Yi_mark = [ '0' if i>0 else '^' for i in Y ]

ax.scatter(zs[Y==1], ys[Y==1], color="r")
ax.scatter(xs[Y==0], ys[Y==0], color="b")

#ax.scatter(xs, ys,c=Yi_color, marker=Yi_mark )#

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

plt.show()


#%% test in disabled group

Xfit= X[ np.ix_(df_filter['group'] == 1, features_P3 ) ]   #range(1,496) extract X data for training from subject s_i and features P300 network
Yfit= Y[ df_filter['group'] == 1]  

rng = np.random.RandomState(0)

Xi_fit,Yi_fit = shuffle(Xfit, Yfit, random_state=rng)

# Xfit= X[ np.ix_(df_filter['id'] == '8', features_P3 ) ]   #range(1,496) extract X data for training from subject s_i and features P300 network
# Yfit= Y[ df_filter['id'] == '8' ]  

# rng = np.random.RandomState(0)

# Xi_fit,Yi_fit = shuffle(Xfit, Yfit, random_state=rng)


clf.fit(Xi_fit,Yi_fit)

sc_test={'s_i':[],
    'accuracy':[],
    'f1':[],
    'roc_auc':[]}

for i, s_i in enumerate( np.unique(df_filter['id'])[:n] ):
    # Select all subjects until n
    
    Xtest= X[ np.ix_(df_filter['id'] == s_i, features_P3 ) ]   # features_P3
    Ytest= Y[ df_filter['id'] == s_i] 
    
    rng = np.random.RandomState(0)

    Xtest,Ytest = shuffle(Xtest,Ytest, random_state=rng)
    
    print( "Subject %s acc: %.4f" %  (s_i, clf.score(Xtest,Ytest) ) ) 
    
    Ypred=clf.predict(Xtest)
    
    sc_test['s_i'].append(s_i)
    sc_test['accuracy'].append( np.round(metrics.accuracy_score(Ytest,Ypred),4) )
    sc_test['f1'].append(np.round(	metrics.f1_score(Ytest,Ypred) , 4))
    sc_test['roc_auc'].append(np.round(	metrics.roc_auc_score(Ytest,Ypred) ,4))
    
    
    test_fpr, test_tpr, te_thresholds = roc_curve(Ytest, clf.predict(Xtest))
    
    #RocCurveDisplay.from_predictions(Ytest, clf.predict(Xtest))
    
    plt.grid()
    
    plt.plot(test_fpr, test_tpr, label='S_'+s_i+" AUC TEST ="+str(auc(test_fpr, test_tpr) )[:4] )
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC(ROC curve)")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.show()

#%% Create table results TRAINING CV

# params table results Xfit s='9';c=1; kfold 20; sig99.0; 
# # params results fitting group Xfit group 1;c=0.5; kfold 10; sig97_5.0; FINAL

df = pd.DataFrame( np.transpose( np.array(list(sc_tot.values())[1:]) ), 
                  columns= list(sc_tot.keys())[1:])

met_2boxplot=list(sc.keys())[1:]

# Show metrics accuracy for each subject in training (cv) [CONTROL GROUP]
# Save table in xlsx
tab_c3= {}


for n,i in enumerate(met_2boxplot):
    
    print(i)
    sc_i=sc[i]
    bx={}
    
    for j in list(np.unique(sc['s_i'])):
        val= list(compress(sc_i,np.array([ j in x for x in sc['s_i']])))
        bx[j]=val
        
    props = dict(boxes="royalblue", whiskers="Black", medians="Black", caps="Black")  
    
    fig, ax = plt.subplots()
    
    pd.DataFrame.from_dict(bx).boxplot(color=props, patch_artist=True, ax=ax)
    
    plt.title(' Training cv '+ i)
    plt.xlabel('Subjects Control Group')
    plt.ylabel(i)
    
    plt.show()
    
    #plt.savefig( path['out']+'Training_cv_'+ i +".jpg",bbox_inches='tight', dpi=500)
        
    pd.DataFrame.from_dict(bx).to_excel(path['out']+'Training_performance_'+i+'.xlsx', 
                                        sheet_name='Sheet'+str(n), index=True)
    
    
    tab_c3[i]= np.array(pd.DataFrame.from_dict(bx).mean())
    tab_c3[i+' sd']=np.array(pd.DataFrame.from_dict(bx).std())
    
pd.DataFrame.from_dict(tab_c3).to_excel(path['out']+'Training_cv_performance_mean_sd.xlsx', index=True)

    
#%% Create table results TEST

df_test = pd.DataFrame( np.transpose( np.array(list(sc_test.values())[1:]) ), 
                  columns= list(sc_test.keys())[1:])

sctest_2barplot=list(sc_test.keys())[1:]



tab_c3_test= {}

for n,i in enumerate(sctest_2barplot):
    print(i)
    sc_i=sc_test[i]
    bx={}
    
    for j in list(np.unique(sc_test['s_i'])):
        val= list(compress(sc_i,np.array([ j in x for x in sc_test['s_i']])))
        bx[j]=val
    
    tab_c3_test[i]= np.array(pd.DataFrame.from_dict(bx).mean())
    #tab_c3[i+' sd']=np.array(pd.DataFrame.from_dict(bx).std())



pd.DataFrame.from_dict(tab_c3_test).plot.bar(color={'accuracy':'#4169E1',
                                                    'f1': '#869EFF',
                                                    'roc_auc':'#C6D7FF'})
plt.ylim(0.5,1)
plt.legend(loc='lower right')
plt.grid(color='grey', linestyle='dashed',linewidth = 1, alpha = 0.25)
plt.tick_params(axis='x', rotation=0)
plt.xlabel('Subjects Disabled Group')


plt.savefig( path['out']+'Test_performance_subjects.jpg',bbox_inches='tight', dpi=500)

pd.DataFrame.from_dict(tab_c3_test).to_excel(path['out']+'Test_performance.xlsx', index=True)

