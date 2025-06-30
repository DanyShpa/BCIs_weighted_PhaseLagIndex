
clc
clear all
close all

addpath('matlab_scripts/f_PlotEEG_BrainNetwork/')

thres=1;

% Sets the paths.
config.data  = '../derivatives/connectivity/individual/conn_allLinks/';
config.out = sprintf('%s%d_perc/','../derivatives/connectivity/individual/figs/conn_threshold_',thres*100);
config.patt = '*.mat';

% Creates the output folder, if required
if ~exist ( fileparts ( config.out ), 'dir' ), mkdir ( fileparts ( config.out ) ); end

% Lists the files.
files = dir ( sprintf ( '%s%s', config.data, config.patt ) );

conds={'con_res_P3', 'con_res_nP3'}
conn_idx={'plv','ciplv','pli','wpli'}
labels={'P300','noP300'}

for i = 1:numel(files)
sprintf('Working subj: %s', files(i).name)    
s = load( sprintf('%s/%s', files(i).folder, files(i).name));

    for k = 1:numel(conn_idx)
     sprintf('Conn Index: %s',conn_idx{k})   
        for j= 1:numel(conds)   
            
        % lets load several networks to be plotted
        figure
        sprintf('Condition: %s',conds{j})
        aij32 = s.(conds{j}).(conn_idx{k}) ;    % 32 channels 

        %% plotting the available layouts
        %f_PlotEEG_BrainNetwork(32)

        %% plotting brain network with nodes of equal sizes and no colorbars  
        % % Here we want to threshold a network of 32 channels for fast plotting
        % under different option to show the links' weights.
        nch = 32; %take care of this variable (nch must be according to matrix size you want to plot it)
        p = thres;   %proportion of weigthed links to keep for.
        aij = threshold_proportional(aij32, p); %thresholding networks due to proportion p
        ijw = adj2edgeL(tril(aij));             %passing from matrix form to edge list form

        logi_p3=load('logi_P300.mat')
        logi_nP3=load('logi_nP300.mat')
        
        logi_p3=logi_p3.logi_p3.aij>0
        
        h=imagesc(aij32.*logi_p3)
        cm2 = colormap(hot);
        cm2 = flipud(cm2);              %inverting colorbar axis
        colormap(cm2)
        cb2 = colorbar('Location', 'eastoutside', 'fontsize', 15);
        caxis([0 1])
        xticks_idx=[1 5 10 15 20 25 30];
        xlabs= cellstr(s.ch_names)
        xticks(xticks_idx)
        yticks(xticks_idx+2)
        xticklabels(xlabs(xticks_idx))
        yticklabels(flip(xlabs(xticks_idx)))
        box off

        %clim([0 0.4])
        
        n_features = sum(aij, 2);
        cbtype = 'wcb';

        figure(j);
        f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx', n_features, 'n_nn2nx', cbtype);
        
        title(gca,sprintf('%s',labels{j}),'FontSize',18)
        
        
        new_dir=[config.out,conn_idx{k},'/']
        
        if ~exist ( fileparts ( new_dir ), 'dir' ), mkdir ( fileparts ( new_dir ) ); end%       
  
        filename =sprintf('%s/Subj_%s_conn_%s_partial.jpg',new_dir,s.id,conn_idx{k});
        print(figure(1), filename, '-dpng', '-r600')
        
        close all
        
        end

        %First Figure 
        figure(1)
        h1 = gca; % get handle to axes of figure
        %Second Figure
        figure(2)
        h2 = gca;
        
        %create new figure        
        h3 = figure(3);
        
        txt = sprintf('Index: %s',conn_idx{k})
        text(0.38,0.99,txt, 'FontSize', 20, 'FontWeight', 'bold')
        
        axis off;
        box off;
        
        h1b = copyobj(h1,h3); %copy children to new parent axes i.e. the subplot axes
        h2b =  copyobj(h2,h3);
        
        set( h1b, 'Position', [0.03 0.0 0.4 0.9])
        set( h2b, 'Position', [0.45 0.0 0.4 0.9])
        
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
        
        set( h1b, 'Title', get(h1,'Title'))
        set( h2b, 'Title', get(h2,'Title')) 

        cm2 = colormap(hot);
        cm2 = flipud(cm2); 
        colormap(cm2)
        colorbar('Location', 'eastoutside', 'fontsize', 15, 'FontWeight', 'bold');
        caxis([0.5, 1])
        %clim([0.5, 1])

%         ax_tit = axes('Position',[0.4 0.9 0.2 0.1]);
%         axis off;
%         

%         title('Test',sprintf('Index: %s',conn_idx{k}) ,'FontSize',18)
%         set(get(gca,'title'),'Position',[5.5 0.4 0.99])

        new_dir=[config.out,conn_idx{k},'/']
        
        if ~exist ( fileparts ( new_dir ), 'dir' ), mkdir ( fileparts ( new_dir ) ); end%       

        saveas(figure(3),sprintf('%s/Subj_%s_conn_%s.jpg',new_dir,s.id,conn_idx{k}))
        close all
        
    end
end
