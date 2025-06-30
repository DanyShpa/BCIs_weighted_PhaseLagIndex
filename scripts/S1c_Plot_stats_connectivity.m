
clc
clear all
close all

%addpath('matlab_scripts/f_PlotEEG_BrainNetwork/')


% Sets the paths.
config.data  = '../derivatives/connectivity/group/stats/';
config.out = '../derivatives/connectivity/group/stats/figs/';
config.patt = '*.mat';

% Creates the output folder, if required
if ~exist ( fileparts ( config.out ), 'dir' ), mkdir ( fileparts ( config.out ) ); end

stats_data = load([config.data,'stats_all.mat'])

%% Join hemispheres events

f_names=fieldnames(stats_data);

H={}

H.(f_names{1})= str2num(stats_data.(f_names{1}))
H.(f_names{2})= stats_data.(f_names{2})'
H.(f_names{3})= cellstr(stats_data.(f_names{3}))
H.(f_names{4})= stats_data.(f_names{4})'
H.(f_names{5})= stats_data.(f_names{5})'
H.(f_names{6})= stats_data.(f_names{6})'
H.(f_names{7})= stats_data.(f_names{7})'
H.(f_names{8})= stats_data.(f_names{8})'
H.(f_names{9})= stats_data.(f_names{9})'
H.(f_names{10})= stats_data.(f_names{10})'


H = struct2table(H);

writetable(H,'H.xlsx');
% {'id'                  }
%     {'group'               }
%     {'erp'                 }
%     {'thres'               }
%     {'mean_conn'           }
%     {'sd_conn'             }
%     {'n_conn_sigLks'       }
%     {'degree_cen_mean'     }
%     {'closeness_cen_mean'  }
%     {'betweenness_cen_mean'}
%     {'Properties'          }
%     {'Row'                 }
%     {'Variables'           }

%% Subject mean sd; t-test plv sig links

H.id_label = categorical(H.id);
H.thres_label = categorical(H.thres);
H.erp_label = categorical(H.erp);
H.group_label = categorical(H.group);


features={'mean_conn','n_conn_sigLks','degree_cen_mean',...
    'closeness_cen_mean','betweenness_cen_mean'}

H.joint_labels=categorical(H.group_label.*H.thres_label);%.*H.erp_label
group_plot_lab={'Disabled','Control'}

cat_labs=categories(H.joint_labels)

j_0=0    
    for j= 1:numel(categories(H.joint_labels))
        lab_j=cat_labs(j)
        
        k_0=0
        for k= 1:numel(features)
            feat=features{k};
            
            colmn=strcmp(feat,fieldnames(H));

            x= double(H{ H.joint_labels == lab_j & H.erp_label=='P300',colmn});
            y= double(H{ H.joint_labels == lab_j & H.erp_label=='nP300',colmn});

            %[h,p,ci,stats] = ttest2(x,y,'Tail','left'); % , 'Vartype','unequal',);,'Tail','left'
            [p,h,stats] = ranksum(x,y)
            
            test(j+j_0+k_0).thres=string(lab_j);
            test(j+j_0+k_0).feature=feat;
            test(j+j_0+k_0).mean_P3=mean(x);
            test(j+j_0+k_0).sd_P3=std(x);
            test(j+j_0+k_0).mean_nP3=mean(y);
            test(j+j_0+k_0).sd_nP3=std(y);
            test(j+j_0+k_0).h=h;
            test(j+j_0+k_0).p=p;
            test(j+j_0+k_0).stats_ranksum=stats.ranksum
            %test(j+j_0+k_0).ci=ci;
            %test(j+j_0+k_0).stats_tstat=stats.tstat;
            %test(j+j_0+k_0).stats_df=stats.df;
            %test(j+j_0+k_0).stats_sd=stats.sd;
            
            k_0=k_0+1

        end
        
    j_0=j_0+k_0 -1    
    
    end
filename = sprintf('%s../%s',config.out,'stats_table.xlsx') ;   

%save(filename,'test')
table_stats_test=struct2table(test);

writetable(table_stats_test,filename);


% %% Plot by subject
% 
% features= H.Properties.VariableNames([5 7:10]);
% features_title={'wpli sig links', 'number sig links', 'centrality degree',...
%     'centrality closeness', 'centrality betweeness'}
% thres= categories(categorical(H.thres))
% for t = 1: numel( thres)
%     
%     thres_i=thres{t}
%     
%     
% 
%     for i= 1:numel(features)
%         
%         my_H=H(H.thres_label==thres_i,:)
% 
%         my_H=my_H(:,["id_label", "erp_label",features{i}])
% 
%         X= 1:numel(categories(H.id_label))%str2double(categories(H.id_label))
%         Y= reshape(my_H.(features{i}), 2,[])'
% 
%         figure
%         set(gcf, 'Position', [380,258,1250,500])
%         b=bar(X,Y) 
%         
%         b(1).FaceColor='r'
%         b(2).FaceColor='b'
% 
%         legend('P300','noP300')
%         ylim([0 max(Y,[], 'all')+max(Y,[], 'all')/10])
%         
%         ax = gca;
%         ax.FontSize = 17;
%         ax.FontWeight = 'bold';
%         
%         grid on
%         ax.GridAlpha  = 0.7
%         ax.GridLineStyle='--'
%         
%         title(features_title{i},  'fontsize', 20 )%'interpreter', 'latex',
%         xlabel('Subjects')
%         ylabel('Mean' )
% 
%         % Creates the output folder, if required
%         th_save=replace(thres_i, '.','_')
%         
%         if ~exist ( [config.out,'thres_',th_save] , 'dir' )
%             mkdir ( [config.out,'thres_', th_save]  ); 
%         end
%         %saveas(figure(1),sprintf('%s/%s_%s_events_boxplot.eps',config.path.out,features{i},freq_band),'epsc')
% 
%         saveas(gcf,sprintf('%sthres_%s/%s_barplot.jpg',config.out,th_save,features{i}) )
% 
% %
%     end
% 
%     close all
%     
% end