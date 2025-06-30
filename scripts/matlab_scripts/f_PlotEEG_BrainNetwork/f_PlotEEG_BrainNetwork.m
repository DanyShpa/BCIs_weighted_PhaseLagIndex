function [] = f_PlotEEG_BrainNetwork(nch, ijw, w_scal, n_features, n_scal, cbtype)
% 
% % f_PlotEEG_BrainNetwork.m plots a brain functional network based on 4 EEG
% layouts of (32, 56, 61 and 64) channels and different options according
% the user needs.
% 
% USAGE.
% f_PlotEEG_BrainNetwork(nch)
%           Shows the chosen layout
% f_PlotEEG_BrainNetwork(nch, ijw, w_scal)
%           Shows brain network with nodes of equal sizes and no colorbars 
% f_PlotEEG_BrainNetwork(nch, ijw, w_scal, n_features, n_scal, cbtype)
%           Shows brain network with nodes of different sizes and colorbars
%           if the user wants. 
% 
% INPUTS.
% NCH: Number of channels to plot (it defines the layout to plot)
% IJW: Edge list form of a graph (source(i) -> target(j): weight(w)). The
%           rows in this variable indicate the number of links.
% W_SCAL: Type of scales the plot weights. The size and color of line
%           (link) is proportional to the respective weight. The weighter
%           the link, the dark and wider the line. Options can be:
%           'w_intact': Lines to plot weights remain intact as the third column of ijw
%           'w_unity': Lines to plot weights span from the minimum weight to ONE
%           'w_wn2wx': Lines to plot weights span from a minimum weight
%           (wn) to maximum weigth (wx). Default values of limits are 1 and
%           10. You can internally modify them as you wish 
% N_FEATURES: Node features. They can be Strength, Clustering, etc
% N_SCAL: Type of scales to plot nodes attributes as circles. The radious
%           of a circle (node) is  proportional to the node's attribute and
%           according to the colormap JET. The more attribute a node has,
%           the bigger and hotter the node is plotted. Options can be:
%           'n_fixed': Nodes sizes are fixed in the inner variable
%           FIXEDSIZENODE (default = 2). This option makes no colorbar
%           being ploted.
%           'n_nn2nx': Nodes sizes are plotted in circles that span from a
%           minimum attribute (nn) to maximum one (nx). Default values of
%           limits are 1.5 and 3.5. You can internally modify them as you wish
% CBTYPE: Type of colorbar to be showed in the figure. Options can be:
%           'nocb': NO colorbar to be displayed (default for n_scal = 'n_fixed')
%           'wcb':  Weigth colorbar in a colormap BONE plotted in left side
%           'ncb':  Node colorbar in a colormap JET plotted in right side
%           'wncb': Weigth and Node colorbars in both colormapscool cool and cool
% 
% OUTPUTS.
% Figure of a head, nodes, links, building the brain network and colorbars.
% 
% % NOTE 1. This function only admits the aforementioned layouts based on
%           these set of nodes: 32, 56, 61 and 64. In future releases, if
%           you need another kind of layout, you can modify this code in
%           the section: ASSOCIATING CHANNELS & POSITIONS DUE TO TEMPLATE.
% % NOTE 2. Always be sure that nch variable correspond to the layout you
%           want plot and associate with the data stored in ijw.
% % NOTE 3. The more links you take, the much time this function speend for
%           plotting the network. Examples of machine time to plot full
%           connected networks with to colorbars using 'w_wn2wx' and
%           'n_nn2nx' as options:
%           32 nodes => t = 22 sec
%           56 nodes => t = 208 sec
%           61 nodes => t = 387 sec
%           66 nodes => t = 396 sec
% ..............................................................................
% 
% JohannM.
% Bogotá (2016)
% johemart@gmail.com


% % % % % ............................. FOR TESTING PURPOSES ...........................
% % % % % ..............................................................................
% % % % clear all;  clc;    close all;
% % % % aij32 = importdata('aij32.txt');    aij56 = importdata('aij56.txt');    %testing matrices
% % % % aij61 = importdata('aij61.txt');    aij64 = importdata('aij64.txt');
% % % % aij = threshold_proportional(aij32, .03);    %thresholding networks due to proportion p. p=1 (all weights preserved)
% % % % 
% % % % nch = 32;                       %take care of this (NCH must be according to matrix size)
% % % % 
% % % % ijw = adj2edgeL(triu(aij));
% % % % w_scal = 'w_wn2wx';             % more options: 'w_intact', 'w_unity', 'w_wn2wx'
% % % % 
% % % % n_features = sum(aij, 2);       % in this case the feature is the Strenght
% % % % n_scal = 'n_nn2nx';             % more options: 'n_fixed', 'n_nn2nx'
% % % % cbtype = 'wncb';                % more options: 'wcb', 'wncb'
% % % % 
% % % % nargin = 6;                     % testing the number of arguments of input
% % % % % ............................. FOR TESTING PURPOSES ...........................
% % % % % ..............................................................................


% % % % % % % % % % % % % % % % % CODIFYING THE WAY TO SHOW THE RESULTING FIGURE
if nargin == 1                          %Number of channels
    % disp(nch);                          %Displaying number of nodes (activate it for testing purposes)
    fplot_layoutEEG(nch);               %Showing its associated layout (deactivate it for testing purposes)   
    return;
elseif nargin == 3                      %Previous + Matrix and the Scaling of Weights
    w_atribut = ijw(:, 3);              %taking the weights' attributes
    w_scaling = w_scal;                 %taking the scaling of weigths
    n_atribut = ones(nch, 1);           %fixing nodes' attributes
    n_scaling = 'n_fixed';              %fixing nodes' scaling
    cbnet = 'nocb';                     %fixing not to show the colorbars
elseif nargin == 6                      %Previous + Nodes' attributes and Node Scaling
    w_atribut = ijw(:, 3);              
    w_scaling = w_scal;
    n_atribut = n_features;             %taking nodes attributes
    
    n_scaling = n_scal;                 %condition of nodes' attributes and colorbars
    if strcmp(n_scaling, 'n_fixed')
        cbnet = 'nocb';                 %fixing not to show the colorbars
    elseif strcmp(n_scaling, 'n_nn2nx')
        cbnet = cbtype;                 %taking the scaling of nodes
    end
else
    warning('You must insert 1, or 3, or 6 inputs');
    error('You are inserting wrong number of inputs');
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % % % % ASSOCIATING CHANNELS & POSITIONS DUE TO TEMPLATE
% % % Original Layout from FieldTrip of BrainCapMR 66 total channels.
% EasycapM11-61-Channel-Arrangement (used in BrainCap64). EasyCap Enterprises.
% For 64 + (ground) + (QRS) Montage (complete montage including ground and heart)
% 64 channels: 61 + 3 periferal channels Montage
% 61 channels Montage
% 56 channels Montage
% 32 channels Montage
% (Contain, XYZ electrode positions, labels head shapes)
load easycapM11.mat; %(lay struct) provided from EasyCap Enterprises.

load labelsEEG.mat; %cell array of 4 desired layouts of nodes (made by myself)

nchannels = nch;                % number of channels defines the position layout 
if nchannels == 32
    labels = labelsEEG{4};      %labels of a 32 montage
elseif nchannels == 56
    labels = labelsEEG{2};      %labels of a 56 montage
elseif nchannels == 61
    labels = labelsEEG{3};      %labels of a 61 montage
elseif nchannels == 64
    labels = labelsEEG{4};      %labels of a 64 montage
else
    warning('Type the correct number of channels: 32,56,61,64')
    error('Here are just 4 layouts for 32, 56, 61 and 64 channels in EaisyCap Enterprises');
end
% % % comparing desired labels with original general labels
[~,b] = (ismember(labels, lay.label));   %b are the desired nodes' indexes from the gral layout
r_nodepos = lay.pos(b, :);      %position of desired nodes depending on the layout
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % % % SCALING the LINKS' WEIGHTS in three different ways
switch w_scaling
    case 'w_intact'   % keep weights intact
        w_atribut_new = w_atribut;
    case 'w_unity'    % scaling weights to unity
        w_atribut_new = w_atribut./max(w_atribut);
    case 'w_wn2wx'     % scaling between wn to wx (default wn=1, wx=10)
        wn = 1;
        wx = 10;%10
        %w_atribut_new = wn + (((w_atribut - min(w_atribut))*(wx-wn))/(max(w_atribut) - min(w_atribut)));
        w_atribut_new = w_atribut./max(w_atribut);
        
        if sum(w_atribut) ==0
            w_atribut_new = w_atribut;
        end
           
    otherwise
        error('type w_scaling as: ''w_intact'' or ''w_unity'' or ''w_wn2wx''')
end
% % % associating weight's attributes to inversed colormap cool
% RGBlinks = squeeze(double2rgb(w_atribut_new, flipud(colormap(hot)) ));
% 
% pos_links=find(w_atribut_new);
% colors_links_nonzero= squeeze(double2rgb(w_atribut_new(pos_links), flipud(colormap(hot)) ))
% 
% RGBlinks(pos_links,:)=colors_links_nonzero
RGBlinks = squeeze(double2rgb(w_atribut_new, flipud(colormap(hot)) ));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % %  SCALING the NODES'S ATRIBUTES in two different ways
switch n_scaling
    case 'n_fixed'
        fixedsizenode = 2;
        n_atribut_new = n_atribut.*0 + fixedsizenode;
        % % % associating equal nodes' attributes to ONE color
        RGBnodes = squeeze(double2rgb(n_atribut_new, colormap([210,180,28]./255)));
    case 'n_nn2nx'          % scaling between nn to nx (default nn=1, nx=10)
        nn = 1;%1
        nx = 1;%10
        n_atribut_new = nn + (((n_atribut - min(n_atribut))*(nx-nn))/(max(n_atribut) - min(n_atribut)));
        % % % associating nodes' attributes to colormap hot
        RGBnodes = squeeze(double2rgb(n_atribut_new, colormap(hot)));
        
    otherwise
        error('type n_scaling as: ''n_fixed'' or ''n_nn2nx''')
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % %  STARTING TO DISPLAY THE FIGURE  % % % % % % % % % %
axis square;                                    %making it squared
axis off;                                       %avoiding the axis
% % % Put here the line (EXPANDING FIGURE ON SCREEN) if you want to see an
% animation version when activate the line (PAUSE(.1))
% % % :::::::::::::::::::::::::::::: PLOTTING HEAD :::::::::::::::::::::::::::
hold on;
% % % plotting head layout
plot(lay.outline{1}(:,1), lay.outline{1}(:,2), 'k', 'LineWidth',4)
plot(lay.outline{1}(:,1), lay.outline{1}(:,2), 'g', 'LineWidth',2, 'color', [119,136,153]./255)

plot(lay.outline{2}(:,1), lay.outline{2}(:,2), '--k', 'LineWidth',4)
plot(lay.outline{2}(:,1), lay.outline{2}(:,2), 'g', 'LineWidth',2, 'color', [119,136,153]./255)

plot(lay.outline{3}(:,1), lay.outline{3}(:,2), '--k', 'LineWidth',4)
plot(lay.outline{3}(:,1), lay.outline{3}(:,2), 'g', 'LineWidth',2, 'color', [119,136,153]./255)

plot(lay.outline{4}(:,1), lay.outline{4}(:,2), '--k', 'LineWidth',4)
plot(lay.outline{4}(:,1), lay.outline{4}(:,2), 'g', 'LineWidth',2, 'color', [119,136,153]./255)

% % % plotting links and nodes
for lk = 1 : size(ijw, 1)           %along all links
    xynodes = zeros(2, 2);          %choose the position XY of nodes of one link
    
    for nd = 1 : 2                                  %for the two nodes of a link
        xynodes(nd, :) = r_nodepos(ijw(lk,nd), :);  %hold the positions
    end
    
    if ijw(lk, 3)== 0
        line(xynodes(:, 1), xynodes(:, 2), 'LineWidth',1, 'Color', 'none');
    else
        line(xynodes(:, 1), xynodes(:, 2), 'LineWidth',6 , 'Color', RGBlinks(lk, :));% ijw(lk, 3) stable size links
    end
    %line(xynodes(:, 1), xynodes(:, 2), 'LineWidth', w_atribut_new(lk), 'Color', RGBlinks(lk, :));
    % % %     line(xynodes(:, 1), xynodes(:, 2), 'LineWidth', ijw(lk, 3), 'Color', [.8 .8 .1]); % links de un solo color
    
    for nd = 1 : 2                                  %for the two nodes of a link
        plot(xynodes(nd, 1), xynodes(nd, 2), 'o', 'MarkerSize', 8*pi, ...
            'MarkerEdgeColor',[0.7 0.7 0.7], 'MarkerFaceColor', "white", 'LineWidth',1);
%         plot(xynodes(nd, 1), xynodes(nd, 2), 'o', 'MarkerSize', 4*pi.*n_atribut_new(ijw(lk,nd)), ...
%             'MarkerEdgeColor','k', 'MarkerFaceColor', RGBnodes(ijw(lk,nd), :), 'LineWidth',1);
%         
        text(xynodes(nd, 1), xynodes(nd, 2), labels{ijw(lk,nd)}, 'fontsize', 10, 'FontWeight','bold', 'HorizontalAlign', 'center');
%         text(xynodes(nd, 1)-0.03, xynodes(nd, 2), labels{ijw(lk,nd)}, 'fontsize', 12, 'FontWeight','bold');
        %text(xynodes(nd, 1)+0.01, xynodes(nd, 2)+0.01, num2str(ijw(lk,nd)), 'fontsize', 15); % number of node as label
    end
    %         pause(.1)
end
set(gca,'XTick',[],'YTick',[]);     %avoiding the tick labels of axis
axis tight;                         %setting axis limits to the range of data
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % % % % %  CODIFYING THE DOUBLE COLORMAPS AND COLORBARS
ax1 = gca;                          %getting current axis of figure
if strcmp(cbnet, 'nocb')
    display('Nodes scaling is fixed')
elseif strcmp(cbnet, 'wcb')
    % % % % LINKS
    cm2 = colormap(hot);
    cm2 = flipud(cm2);              %inverting colorbar axis
    colormap(cm2)
    cb2 = colorbar('Location', 'eastoutside', 'fontsize', 20);
    %caxis([0.5 max(w_atribut_new)]);%min(w_atribut_new)
    %caxis([min(w_atribut_new(pos_links)) max(w_atribut_new(pos_links))]);
    cmin = 0.5;
    cmax = max(w_atribut_new);

    if cmax > cmin
        caxis([cmin cmax]);
    else
        caxis([0 cmin]);  % or skip caxis entirely
    end

    
    %cbfreeze(cm2)                   %freezing this colormap
    %freezeColors(ax1);               %freezing this colorbar
    
elseif strcmp(cbnet, 'ncb')
    % % % % NODOS
    cm1 = colormap(hot);
    cb1 = colorbar('Location', 'east', 'fontsize', 30, 'FontWeight','bold');
    caxis([0.5 max(n_atribut)]);%taking the nodes' attributes different from zero min(n_atribut(n_atribut>0))
    cbfreeze(cm1)
    freezeColors(ax1)
elseif strcmp(cbnet, 'wncb')
    % % % % NODOS
    cm1 = colormap(hot);
    cb1 = colorbar('Location', 'east', 'fontsize', 30);
    clim([0.5 1])
    caxis([0.5 max(n_atribut)])
    cbfreeze(cm1)
    freezeColors(ax1)
    % % % % LINKS
    cm2 = colormap(hot);
    cm2 = flipud(cm2);
    colormap(cm2);
    cb2 = colorbar('Location', 'west', 'fontsize', 30);
    clim([0.5 1])
    caxis([0.5 max(w_atribut)])%min(w_atribut)
    cbfreeze(cm2)
    freezeColors
else
    error('type cbnet as: ''nocb'', ''wcb'', ''ncb'', ''wncb''');
end
hold off;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]) %EXPANDING FIGURE ON SCREEN



% % % ..........................................................................
% % % ..........................................................................
% % % DEACTIVATE THIS CHILD FUNCTION IF YOU WANT TO TEST THE INNER CODE OF
% THE PARTN FUNCTION f_PlotEEG_BrainNetwork
function [] = fplot_layoutEEG(nch)

% % % ..........................................................................
% % Defining the nodes' layouts
load easycapM11.mat; %(lay struct) provided from EasyCap Enterprises.
load labelsEEG.mat; %cell array of 4 desired layouts of nodes (made by myself)
nchannels = nch;                % number of channels defines the position layout 
if nchannels == 32
    labels = labelsEEG{4};      %labels of a 32 montage
elseif nchannels == 56
    labels = labelsEEG{2};      %labels of a 56 montage
elseif nchannels == 61
    labels = labelsEEG{3};      %labels of a 61 montage
elseif nchannels == 64
    labels = labelsEEG{4};      %labels of a 64 montage
else
    warning('Type the correct number of channels: 32,56,61,64')
    error('Here are just 4 layouts for 32, 56, 61 and 64 channels in EaisyCap Enterprises');
end
% % % comparing desired labels with original general labels
[~,b] = (ismember(labels, lay.label));   %b are the desired nodes' indexes from the gral layout
r_nodepos = lay.pos(b, :);      %position of desired nodes depending on the layout

% % % ..........................................................................
% % % % % % % % % % % % % perfect head of all 32 channels
hold on
plot(lay.outline{1}(:,1), lay.outline{1}(:,2), 'k', 'LineWidth',3)
plot(lay.outline{1}(:,1), lay.outline{1}(:,2), 'g', 'LineWidth',1, 'color', [222,184,135]./255)
plot(lay.outline{2}(:,1), lay.outline{2}(:,2), '--k', 'LineWidth',3)
plot(lay.outline{2}(:,1), lay.outline{2}(:,2), 'g', 'LineWidth',1, 'color', [222,184,135]./255)
plot(lay.outline{3}(:,1), lay.outline{3}(:,2), '--k', 'LineWidth',3)
plot(lay.outline{3}(:,1), lay.outline{3}(:,2), 'g', 'LineWidth',1, 'color', [222,184,135]./255)
plot(lay.outline{4}(:,1), lay.outline{4}(:,2), '--k', 'LineWidth',3)
plot(lay.outline{4}(:,1), lay.outline{4}(:,2), 'g', 'LineWidth',1, 'color', [222,184,135]./255)
for ch = 1 : nch
    plot(r_nodepos(ch, 1), r_nodepos(ch, 2), 'o', 'MarkerSize', 20, 'MarkerEdgeColor','k', 'MarkerFaceColor', [240,128,128]./255, 'LineWidth',1.5)
    text(r_nodepos(ch, 1)-0.03, r_nodepos(ch, 2)-0.03, labels(ch), 'fontsize', 15);
    text(r_nodepos(ch, 1)+0.03, r_nodepos(ch, 2)+0.03, num2str(ch),'fontsize', 15); %if you want to plot the node's numbers
    %pause(.05); %YOU CAN DEACTIVATE THIS LINE IF NOT INTEREST TO SEE THE ANIMATION OF LAYOUT
end
axis square;
axis off;
box off;
set(gca,'XTick',[],'YTick',[]);
axis tight;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
hold off;
return