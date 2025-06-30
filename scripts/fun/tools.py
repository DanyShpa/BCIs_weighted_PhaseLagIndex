# -*- coding: utf-8 -*-
# """

# @author: Dany
# """
import scipy.io
import mne
from mne import io
#from mne_connectivity import spectral_connectivity
#from mne_viz import plot_sensors_connectivity,plot_connectivity_circle
import numpy as np
import pylab as plt
from random import seed
from random import choice, sample
import neurotools 
from neurotools.signal.phase import phase_randomize
import mne
from mne import io
from mne_connectivity import spectral_connectivity_epochs
import winsound
from matplotlib.colors import SymLogNorm
import networkx as nx
import neurokit2 as nk # using this one
#Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity
# tests. Physical review letters, 77(4), 635.
#https://neuropsychology.github.io/NeuroKit/functions/signal.html#signal-surrogate
#**IAAFT**: Returns an Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate.
      # It is a phase randomized, amplitude adjusted surrogates that have the same power spectrum
      # (to a very high accuracy) and distribution as the original data, using an iterative scheme



#%% sigLinks
def sigLinks(real, surr, thres, mat_size=32 ):#  numel_surr,
    
    from random import sample
    
    pc=thres; #thres
    
    #real, surr=con_flat_P3,surr_P300
    pvalues=np.zeros(len(real));
    
    for i in range (len(real)):
        
        real_i=real[i];
        
        surr_i= surr[:,i];# numel_surr

        perc_i = np.percentile(surr_i,pc);
        
        if real_i>perc_i:
            
            pvalues[i]=real_i
            #print('real_i=',real_i,'perc_i',perc_i)
            link=i;
            

            n, bins, patches=plt.hist(surr[:,link],density=True,bins=100,color='b', edgecolor='b')
            n = n.astype('int') # it MUST be integer
            # Good old loop. Choose colormap of your taste
            for i in range(len(patches)):
                
                patches[i].set_facecolor(plt.cm.binary(n[i]/max(n)))
            
            plt.axvline(np.percentile(surr[:,link],pc), color='#FF0000', 
                        linestyle='dashed', linewidth=5)
            
            plt.axvline(real[link], color='#FF0000', linestyle='dotted', linewidth=5)
            min_ylim, max_ylim = plt.ylim()
            
            plt.text(np.percentile(surr[:,link],pc )*1.01, max_ylim*0.5,
                     'Surrogate wPLI \n Percentile 95=\n {:.2f}'.format(
                         np.percentile( surr[:,link],pc ) ), rotation=90, 
                          size="large")
            
            plt.text(real[link]*1.01, max_ylim*0.25, 'wPLI original=\n {:.2f}'.format(
                real[link]), rotation=90, 
                 size="large")
            plt.xlabel('wPLI values distribution')
            plt.ylabel('Frequency')
            
            print('abc')
            
            
            
            
        else:
            
            pvalues[i]=0

    #print(sum(pvalues!=0))
    #pval_no0=np.where(pvalues!=0)
    #print(pval_no0)
    
    low_tri_ind = np.tril_indices(mat_size, k = -1) 
    pval_conn_matrix=np.zeros((mat_size,mat_size))
    
    pval_conn_matrix[low_tri_ind[0],low_tri_ind[1]]=pvalues
    
    #%% plot surrogate process
   
    
        
    return pvalues, pval_conn_matrix
      


#%% Surrogates

# Create surrogate connectivity matrices data from raw real time series epoch array
# Randimize the phase and preserve properties of real data (spectra)
def Surrogate_conn_PhaseRandom(x, n, conn_params):
    from random import sample
    
 # Phase surrogate repeat n times
 
    con_flat_tot=np.ndarray(shape=(n,496))
    #random_x=np.empty_like(x)
    
    random_x=x.copy()
    # reduce de number of trials from which generate the n surrogates
    random_x= random_x[sample(range(250), 50),:,: ]#1:250
    
    
    for k in range(n):
       # surrogate
       for i in range(len(random_x[:,1,1])):
           for j in range(len(random_x[1,:,1])):
               
               #np.random.shuffle(random_x[i,j,:])
               
               #random_x[i,j,:] = phase_randomize(x[i,j,:])
               random_x[i,j,:] = nk.signal_surrogate(x[i,j,:],max_iter=5)
               
       winsound.Beep(250, 500)   
       
       con = spectral_connectivity_epochs(
           random_x, conn_params['ch_names'],method=conn_params['conn_methods'],
           mode='multitaper', sfreq=conn_params['sfreq'],fmin=conn_params['fmin'],
           fmax=conn_params['fmax'],faverage=True,tmin=conn_params['tmin'],
           mt_adaptive=False, n_jobs=1)
       
       con = con.get_data(output='dense')[:, :, 0]
       con_flat=con[np.tril_indices(len(conn_params['ch_names']), k = -1)]
       
       winsound.Beep(440, 500)
        
       con_flat_tot[k,:] = con_flat
       print('Repetition number: ', k)
       
        
    # fig, axs = plt.subplots(4)
    # chn_i=12;
    # trial=12;
    
    # t_plot=np.linspace(-200,500,700)
    # y_plot=np.transpose(x[3,chn_i, 0:700]) # pz 12; cz 31
    
    
    
    # # if P300[trial]==True:
    # #     color='#FF0000'
    # # else:
    # #     color='#003fff'
    
    # axs[0].plot(t_plot,y_plot,color='#FF0000')
    # #axs[0].set_ylabel(chanels[chn_i][1],fontsize=12)
    # axs[0].label_outer()
    # axs[0].set_yticks([])
    
    # for surr_i in range(1,4):
        
    #     y_surr_plot=np.transpose(random_x[sample(range(50), 1),chn_i, 0:700]) 
        
    #     axs[surr_i].plot(t_plot,y_surr_plot,color='#808080')
    #     #axs[0].set_ylabel(chanels[chn_i][1],fontsize=12)
    #     axs[surr_i].label_outer()
    #     axs[surr_i].set_yticks([])
    

    
    
    #%% plot compare signal before and after phase randomization
    
    
    fig, ax = plt.subplots()
    fig.tight_layout()
    #for k in range(len(files)):
        

    # Real
    data1=np.mean(x,1)
    x1 = np.arange(data1.shape[1])
    est1 = np.mean(data1, axis=0)
    sd1 = np.std(data1, axis=0)
    cis1 = (est1 - sd1, est1 + sd1)
    
    ax.plot(x1,est1,color='red', lw='1',
            label='Real')
    ax.fill_between(x1,cis1[0],cis1[1],
                    alpha =0.1,
                    label='_Hidden label',
                    lw=0.5,
                    color='red')
    
    # Surrogate
    data=np.mean(random_x,1)
    x2 = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    
    ax.plot(x2,est,color='black', lw='1',
            label='Surrogate')
    
    ax.fill_between(x2,cis[0],cis[1],
                    alpha =0.1,
                    label='_Hidden label',
                    lw=0.5,
                    color='black')
    
    plt.ylim(-0.75,0.75)
    
    a,b = find_pos_t(x2,np.arange(-200,517,1))
    plt.margins(x=0)
    plt.xticks(a, b)
    
    
    
    plt.ylabel('z score')
    #plt.ylabel(r"z-score")
    
    plt.xlabel(r"time (ms)")
    name='Surrogate data P300'
    
    plt.grid(color='grey', linestyle='dashed',linewidth = 1, alpha = 0.25)
    plt.axhline(y=0,color='black', linestyle='dashed',linewidth = 1, alpha = 0.25)
    
    plt.title(name)
    
    plt.legend(loc='upper left') 
    
    # plt.savefig(name+'.png',bbox_inches='tight', dpi=200)
    
    
    plt.show()

    
    
    
    
    
    
    
        
    
   
    

    return con_flat_tot, random_x 

    # con_flat_tot: Matrix_of conn values for surr time series
    # random_x: surr time series


 #%% network measures
def network_centrality(sig_mat, ch_names ):
     
    G = nx.from_numpy_array(sig_mat, create_using=nx.Graph)
    #pos = nx.circular_layout(G)# 
    deg_cent=  dict((ch_names[key], value) 
                    for (key, value) in nx.degree_centrality(G).items())
    deg_cent= dict(sorted(deg_cent.items(), key=lambda item: item[1],reverse=True))
    
    close_cent=  dict((ch_names[key], value) 
                    for (key, value) in nx.closeness_centrality(G).items())
    close_cent=dict(sorted(close_cent.items(), key=lambda item: item[1],reverse=True))
    
    betw_cent=  dict((ch_names[key], value) 
                    for (key, value) in nx.betweenness_centrality(G).items())
    betw_cent=dict(sorted(betw_cent.items(), key=lambda item: item[1],reverse=True))
    
    centrality={'degree':deg_cent,
                'closeness':close_cent,
                'betweenness':betw_cent}
    return centrality
    


# def draw(G, pos, measures, measure_name):
    
#     nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
#                                    node_color=list(measures.values()),
#                                    nodelist=pos.keys())
#     nodes.set_norm(SymLogNorm(linthresh=0.01, linscale=1, base=10))
    
#     # labels = nx.draw_networkx_labels(G, pos)
#     edges = nx.draw_networkx_edges(G, pos)

#     plt.title(measure_name)
#     plt.colorbar(nodes)
#     plt.axis('off')
#     plt.show()
 
    
    
#%% S0_time_series
def clear_all():
    #"""Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
     
        del globals()[var]
        
def find_pos_t(x,t):
    pos=[]
    t_pos=[]
    for num in x:
        if np.equal(np.mod(num/100, 1), 0): # if modulo 1 is 0, then integer
            pos.append(np.where(x==num))
            t_pos.append(t[np.where(num==x)])
            
    y_tick_lab=np.squeeze(np.vstack(np.round(t_pos,1),))
    y_tick_lab[2]=0
    
    return np.squeeze(np.vstack(pos)), y_tick_lab
        
def ERP_baseline_z(x, t):
    
    # Find average value for each epoch at baseline interval
    
    t_a0= (t>=-0.200) & (t<0)
    
    #t_0b= (t>=0) & (t<=0.500)
    
    avg_epoch_baseline = [ np.mean(x[i,:,t_a0[0]], 0) 
                      for i in range(np.shape(x)[0])]
    
    sd_epoch_baseline = [ np.std(x[i,:,t_a0[0]], 0) 
                      for i in range(np.shape(x)[0])]
    # subtract baseline values calculated for each epoch from epoch data
    
    base_x = x 
    
    for i in range(np.shape(x)[0]):# for each trial
        
        for j in range(np.shape(x)[1]):# for each chanel
            
            #for k in range(np.shape(x)[2]):# for each time point
               
            ch_j= avg_epoch_baseline[i][j]
               
            ch_j_sd= sd_epoch_baseline[i][j]
               
            base_x[i,j,:] = (x[i,j,:] - ch_j) / ch_j_sd # each time point of the epoch - avg baseline of that trial and chanel
                
    return base_x

def ERP_baseline(x, t):
    
    # Find average value for each epoch at baseline interval
    
    t_a0= (t>=-0.200) & (t<0)
    
    #t_0b= (t>=0) & (t<=0.500)
    
    avg_epoch_baseline = [ np.mean(x[i,:,t_a0[0]], 0) 
                      for i in range(np.shape(x)[0])]
    
    sd_epoch_baseline = [ np.std(x[i,:,t_a0[0]], 0) 
                      for i in range(np.shape(x)[0])]
    # subtract baseline values calculated for each epoch from epoch data
    
    base_x = x 
    
    for i in range(np.shape(x)[0]):# for each trial
        
        for j in range(np.shape(x)[1]):# for each chanel
            
            #for k in range(np.shape(x)[2]):# for each time point
               
            ch_j= avg_epoch_baseline[i][j]
               
            ch_j_sd= sd_epoch_baseline[i][j]
               
            base_x[i,j,:] = x[i,j,:] - ch_j # each time point of the epoch - avg baseline of that trial and chanel
                
    return base_x
            
def outlier_trials(x,y,f,k):
    # select cut point percentile
    p_low=1
    p_up=100-p_low
        
    # if k==2 or k==3:
    #     p_low=2.5
    #     p_up=100-p_low
        
    mean_trial= np.mean( x, axis=(1,2))
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(10,5))
    
    #plot mean trial hist before correction
    ax0.set_title('Hist mean trial')
    ax0.hist(mean_trial)
    
    outliers_trial= (mean_trial<
                     np.percentile(mean_trial,p_low)) |  (mean_trial>
                                                          np.percentile(mean_trial,p_up))
    
    print(' Percentile ',p_up,' cut point:' , np.percentile(mean_trial,p_up), file=f) 
    print(' Percentile ',p_low,' cut point:', np.percentile(mean_trial,p_low), file=f) 
    
    print('Number of trials deleted total:' + str(np.sum(outliers_trial)) , file=f)
    print('Number of trials deleted P300:' + str(np.sum(y[outliers_trial]==1 )) , file=f)
    
    f.close()
    
    # plot mean trial hist after correction
    ax1.set_title('Hist mean trial NO outliers')
    ax1.hist(mean_trial[ outliers_trial==0 ])
    
    
    
    #return trials array with deleted outlier trials
    
    return x[outliers_trial==0,:,:], y[outliers_trial==0]
    
    
 #%% To delete  

# def tsplot(ax, data,t,**kw):
    
#             x = np.arange(data.shape[1])
#             est = np.mean(data, axis=0)
#             sd = np.std(data, axis=0)
#             cis = (est - sd, est + sd)
            
#             ax.fill_between(x,cis[0],cis[1], **kw)
#             ax.plot(x,est,**kw)
            
#             a,b=find_pos_t(x,t)
#             plt.xticks(a, b)
#             plt.yticks(np.arange(-1.5,1.5,0.5))
#             plt.margins(x=0)
            
#             plt.show()
 
 # def ERP_baseline(x, t):
     
 #     # Find average value for each epoch at baseline interval
     
 #     t_a0= (t>=-0.200) & (t<0)
     
 #     t_0b= (t>=0) & (t<=0.500)
     
 #     avg_epoch_baseline = [ np.mean(x[i,:,t_a0[0]], 0) 
 #                       for i in range(np.shape(x)[0])]
     
 #     # subtract baseline values calculated for each epoch from epoch data
     
 #     base_x = x 
     
 #     for i in range(np.shape(x)[0]):# for each trial
         
 #         for j in range(np.shape(x)[1]):# for each chanel
             
 #             for k in range(np.shape(x)[2]):# for each time point
                
 #                 ch_j= avg_epoch_baseline[i][j]
                    
 #                 if t_0b[0][k]: 
                     
 #                     base_x[i,j,k] = x[i,j,k] - ch_j # each time point of t_0b - avg baseline of thar trial and chanel
                     
 #     return base_x   
 
 # def load_data (path_x,path_y):
 #     ### Epoched data array
 #     mat = scipy.io.loadmat(path_x);
 #     mat.keys()
 #     epochs_data=mat['total_x']
 #     print('epochs_data.shape: epochs x channels x samples ',epochs_data.shape)
 #     print('epochs_data[1,:,:], 1 epoch x 32 channels x 32 samples',epochs_data[1,:,:]) 
     
 #     #Type of epoch (P300/no P300) array
 #     mat = scipy.io.loadmat(path_y);
 #     epoch_type=mat['total_y'][0]
 #     print('epoch_type dims)',epoch_type.shape)
 #     print('epoch_type[1:3]:',epoch_type[1:20])
 #     return epochs_data, epoch_type
    
    
    
    
    
    
    
    
    
    