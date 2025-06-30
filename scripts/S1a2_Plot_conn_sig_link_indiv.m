
clc
clear all
close all

addpath('matlab_scripts/f_PlotEEG_BrainNetwork/')

thres=0.25;

% Sets the paths
config.data  = '../derivatives/connectivity/individual/sig_links_surrogate/';
config.out = sprintf('%s%d_perc/','../derivatives/connectivity/individual/figs/conn_sigLinks_p_0_01/');
config.patt = '*.mat';

% Creates the output folder, if required
if ~exist ( fileparts ( config.out ), 'dir' ), mkdir ( fileparts ( config.out ) ); end

% Lists the files.
files = dir ( sprintf ( '%s%s', config.data, config.patt ) );

labels={'P300','noP300', 'diff (P300-noP300)'}
conds= {'sig_mat_P300','sig_mat_noP300','sig_mat_diff_P300'}

for i = 2:numel(files)
    
sprintf('Working subj: %s', files(i).name)    
s = load( sprintf('%s/%s', files(i).folder, files(i).name));

        for j= 1:numel(conds)   
            
        % lets load several networks to be plotted
         sprintf('Condition: %s',conds{j})
        
        %aij32 = s.(conds{j}).(conn_idx{k}) ;    % 32 channels 
        aij = s.(conds{j})
        
        %% plotting the available layouts
        %f_PlotEEG_BrainNetwork(32)

        %% plotting brain network with nodes of equal sizes and no colorbars  
        % % Here we want to threshold a network of 32 channels for fast plotting
        % under different option to show the links' weights.
        
        nch = 32; %take care of this variable (nch must be according to matrix size you want to plot it)
%         p = thres;   %proportion of weigthed links to keep for.
%         aij = threshold_proportional(aij32, p); %thresholding networks due to proportion p
        ijw = adj2edgeL(tril(aij));             %passing from matrix form to edge list form

        n_features = sum(aij, 2);
        cbtype = 'wcb';

        figure(j);
        
        f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx', n_features, 'n_nn2nx', cbtype);
        
        title(gca,sprintf('%s',labels{j}),'FontSize',18)
        
       
        
        end

        %First Figure P300
        figure(1)
        h1 = gca; % get handle to axes of figure
        
        %Second Figure no P300
        figure(2)
        h2 = gca;
        
        %Second Figure diffs
        figure(3)
        h3 = gca;
        
        %create new figure        
        h4 = figure(4);
        
        txt = sprintf('Subject %s',s.id);
        text(0.40,0.1,txt, 'FontSize', 20, 'FontWeight', 'bold')
        
        axis off;
        box off;
        
        h1b = copyobj(h1,h4); %copy children to new parent axes i.e. the subplot axes
        h2b =  copyobj(h2,h4);
        h3b =  copyobj(h3,h4);
        
        set( h1b, 'Position', [0.0 0.25 0.3 0.6])
        set( h2b, 'Position', [0.32 0.25 0.3 0.6])
        set( h3b, 'Position', [0.64 0.25 0.3 0.6])
        
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
        
        set( h1b, 'Title', get(h1,'Title'))
        set( h2b, 'Title', get(h2,'Title')) 
        set( h3b, 'Title', get(h3,'Title')) 
        

        cm2 = colormap(hot);
        cm2 = flipud(cm2); 
        colormap(cm2)
        cb2 = colorbar('Position', [0.95 0.25 0.01 0.6], 'fontsize', 15, 'FontWeight', 'bold');

%         ax_tit = axes('Position',[0.4 0.9 0.2 0.1]);
%         axis off;
%         box off;

%         title('Test',sprintf('Index: %s',conn_idx{k}) ,'FontSize',18)
%         set(get(gca,'title'),'Position',[5.5 0.4 0.99])

        new_dir=config.out
        if ~exist ( fileparts ( new_dir ), 'dir' ), mkdir ( fileparts ( new_dir ) ); end%       

        saveas(figure(4),sprintf('%s/Subj_%s_sigLinks.jpg',new_dir,s.id))
        close all
        
    
end
