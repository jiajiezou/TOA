# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:41:10 2020

visualizing the results of attention weights prediction

@author: Jiajie
"""
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import  MLPRegressor
from sklearn.linear_model import LinearRegression
import functools
from multiprocessing import Pool
import time
import joblib
import sys
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.image as mpimg
import copy
sys.path.append("..")
import FunctionsZ
import cv2
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from math import sqrt
import cmapy
import os
import math

def patch_gredient(x, y, cmap, ax, fill=True, **kwargs):
    x_len = len(x)
    scat = ax.scatter(
        x, y, s=60, 
        c=np.arange(1, x_len+1), cmap=cmap, 
        **kwargs
        )
    if not fill:
        scat.set_facecolor('none')
    # Create a set of line segments so that we can color them individually
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments, cmap=cmap, alpha=0.8)
    # Set the values used for colormapping
    lc.set_array(np.arange(1, x_len+1))
    lc.set_linewidth(3)
    ax.add_collection(lc)
    
def ContribSpace(
        rgs_result_types, rgs_name, att_seg, global_local_type, clr_map, 
        layer_name, bert_i, bert_name, reason_method, meas
        ):
    doc_ctb_segs = []
    ques_ctb_segs = []
    
    for att_seg_i, att_seg_tmp in enumerate(att_seg):
        doc_ctb_types = []
        ques_ctb_types = []
        
        doc_h = []
        ques_h = []
        for Type_i, Type in enumerate(global_local_type):
            doc_ctb_heads = []
            ques_ctb_heads = []
            
            fig, ax = plt.subplots(
                num=reason_method + bert_name + str(Type_i), 
                figsize=(2.4, 1.8)
                )
            rgs_result = rgs_result_types.loc[
                rgs_result_types.Type.isin(Type), :]
            
            # human & SAR
            doc_h_meas = []
            ques_h_meas = []
            for mea_tmp in meas:
                # doc
                doc_human_perf_l = rgs_result.loc[
                        (rgs_result.target_name==mea_tmp) & 
                        (rgs_result.pca_ratio==1) & 
                        (rgs_result.Fea=='QuesUnreFeaP') & 
                        (rgs_result.rgs_name==rgs_name), 'TestCoef_docs'].tolist()
                
                doc_human_perf_l1 = []
                for doc_human_perf_l_tmp in doc_human_perf_l:
                    doc_human_perf_l1 = doc_human_perf_l1 + doc_human_perf_l_tmp
                doc_human_perf = np.nanmean(doc_human_perf_l1)
                # ques 
                ques_human_perf_l = rgs_result.loc[
                        (rgs_result.target_name==mea_tmp) & 
                        (rgs_result.pca_ratio==1) & 
                        (rgs_result.Fea=='QuesReFea_name') & 
                        (rgs_result.rgs_name==rgs_name), 'TestCoef_docs'].tolist()
                ques_human_perf_l1 = []
                for ques_human_perf_l_tmp in ques_human_perf_l:
                    ques_human_perf_l1 = ques_human_perf_l1 + ques_human_perf_l_tmp
                ques_human_perf = np.nanmean(ques_human_perf_l1)
                if mea_tmp == 'SAR_Atten_lp':
                    ax.scatter(doc_human_perf, ques_human_perf, s=100, c='r', edgecolors='k')
                else:
                    ax.scatter(doc_human_perf, ques_human_perf, s=80, c='k', edgecolors='k')
                doc_h_meas.append(doc_human_perf)
                ques_h_meas.append(ques_human_perf)
            doc_h.append(doc_h_meas)
            ques_h.append(ques_h_meas)
            
            # self attention
            doc_rgs_result = []
            ques_rgs_result = []
            for target_i, target_tmp in enumerate(layer_name):
                head_names = [
                    att_seg_tmp + '-' + str(head_tmp) + '_lp'
                    for head_tmp in range(12*target_tmp, 12*(target_tmp+1))
                    ]
                
                doc_test_coef = []
                for head_name_tmp in head_names:
                    
                    doc_rgs_result_tmp_l = rgs_result.loc[
                        (rgs_result.target_name==head_name_tmp) & 
                        (rgs_result.pca_ratio==1) & 
                        (rgs_result.Fea=='QuesUnreFeaP') & 
                        (rgs_result.rgs_name==rgs_name), 'TestCoef_docs'].tolist()
                    
                    doc_rgs_result_tmp_l1 = []
                    for doc_rgs_result_tmp_l_tmp in doc_rgs_result_tmp_l:
                        doc_rgs_result_tmp_l1 = doc_rgs_result_tmp_l1 + doc_rgs_result_tmp_l_tmp
                    doc_rgs_result_tmp = np.nanmean(doc_rgs_result_tmp_l1)
                    
                    doc_test_coef.append(doc_rgs_result_tmp)
                doc_rgs_result.append(np.mean(doc_test_coef))
                
                ques_test_coef = []
                for head_name_tmp in head_names:
                    ques_rgs_result_tmp_l = rgs_result.loc[
                        (rgs_result.target_name==head_name_tmp) & 
                        (rgs_result.pca_ratio==1) & 
                        (rgs_result.Fea=='QuesReFea_name') & 
                        (rgs_result.rgs_name==rgs_name), 'TestCoef_docs'].tolist()
                    
                    ques_rgs_result_tmp_l1 = []
                    for ques_rgs_result_tmp_l_tmp in ques_rgs_result_tmp_l:
                        ques_rgs_result_tmp_l1 = ques_rgs_result_tmp_l1 + ques_rgs_result_tmp_l_tmp
                    ques_rgs_result_tmp = np.nanmean(ques_rgs_result_tmp_l1)
                    
                    ques_test_coef.append(ques_rgs_result_tmp)
                ques_rgs_result.append(np.mean(ques_test_coef))
                normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
                ax.scatter(doc_test_coef, ques_test_coef, s=10, 
                           c=[target_i/len(layer_name)]*len(ques_test_coef), 
                           cmap=clr_map, norm=normalize, alpha=0.5, edgecolors='')
                
                doc_ctb_heads.extend(doc_test_coef)
                ques_ctb_heads.extend(ques_test_coef)
                plt.pause(0.1)
                
            patch_gredient(doc_rgs_result, ques_rgs_result, cmap=clr_map, ax=ax)
            plt.xlim([-0.05, 0.62])
            plt.ylim([-0.05, 0.42])
            plt.xticks(np.arange(0, 0.7, 0.2))
            plt.yticks(np.arange(0, 0.5, 0.2))
            plt.xlabel('prediction accuracy for bottom-up factors')
            plt.ylabel('prediction accuracy for top-down factors')
            plt.title(bert_name)
            
            plt.subplots_adjust(left=0.25, right=0.95, top=0.88, bottom=0.24, wspace=0.1, hspace=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.pause(0.5)
            
            doc_ctb_types.append(doc_ctb_heads)
            ques_ctb_types.append(ques_ctb_heads)
        
        doc_ctb_segs.append(doc_ctb_types)
        ques_ctb_segs.append(ques_ctb_types)
    return doc_ctb_segs, ques_ctb_segs, doc_h, ques_h
    
def DrawLine(ys, cmap, ax):
    markers = ['+', '.', '*']
    for model_i in range(ys.shape[-1]):
        y_tmp = ys[:, :, model_i]
        patch_gredient(
            np.arange(y_tmp.shape[-1]), y_tmp.mean(axis=0), 
            cmap, ax, fill=True, marker=markers[model_i]
            )
    plt.xticks(np.arange(y_tmp.shape[-1]), np.arange(y_tmp.shape[-1])+1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

#%%Classification
if __name__ == '__main__':
    FunctionsZ.FontSetting()
    regresses_tmp = 'VT_Resi'       # full, VT_Resi
    
    cdict = {'red':   ((0.0,  1.0, 1.0),
                        (0.5,  0.0, 0.0),
                        (1.0,  0.0, 0.0)),
              'green': ((0.0,  0.7, 0.7),
                        (0.5,  0.7, 0.7),
                        (1,  0, 0)),    
              'blue':  ((0.0,  0.0, 0.0),
                        (0.5,  0.0, 0.0),
                        (1,  0.7, 0.7))}
    clr_map = LinearSegmentedColormap("mycmap", cdict, N=512)
    
    reason_methods = ['model_fine', 'model_pre', 'model_random'] 
    bert_models = ['bert', 'albert', 'roberta']
    types = ['Mainly', 'Title', 'GPurpose', 'LPurpose', 'Fact', 'Limply']
    
    for reason_method in reason_methods:
        for bert_tmp in bert_models:
            bert_model = '/' + reason_method + '/' + bert_tmp + '/'
            save_predacc_folder = f'./AttentionMap{bert_model}PredAcc' 
            # -----------------------merge types--------------------#
            rgs_result_types = pd.DataFrame()
            for Type_i, Type in enumerate(types):
                rgs_result = joblib.load(open(
                    save_predacc_folder + 
                    f'pred_atten_{regresses_tmp}.pickle', 'rb'))
                rgs_result['Type'] = Type
                rgs_result_types = rgs_result_types.append(rgs_result)
            joblib.dump(rgs_result_types, open(
                save_predacc_folder + 
                f'pred_atten_{regresses_tmp}_types.pickle', 'wb'))
    
    #--------------------contributions to DNN attention--------------------# 
    BERT_name = ['BERT', 'ALBERT', 'RoBERTa']
    meas = [
        'IA_DWELL_TIME_lp', 
        'IA_FIRST_RUN_DWELL_TIME_lp', 
        'IA_RUN_COUNT_Del0_lp'
        ]
    ques_last_all = []
    for reason_method in reason_methods:
        ques_last_mts = []
        for bert_i, bert_tmp in enumerate(bert_models):
            bert_model = '/' + reason_method + '/' + bert_tmp + '/'
            save_predacc_folder = f'./AttentionMap{bert_model}PredAcc' 
            rgs_result_types = joblib.load(open(
                save_predacc_folder + 
                f'pred_atten_{regresses_tmp}_types.pickle', 'rb'))
            
            # the contributions of text/question related features 
            global_local_type = [['LPurpose', 'Fact', 'Limply'], ['Mainly', 'Title', 'GPurpose']]
            layer_name = [lay_i for lay_i in range(12)]
            rgs_name = 'LR'
            att_seg = ['CLS2P'] 
            doc_ctb_segs, ques_ctb_segs, doc_h, ques_h = ContribSpace(
                rgs_result_types, rgs_name, att_seg, 
                global_local_type, clr_map, layer_name, 
                bert_i, BERT_name[bert_i], reason_method, meas)
            
            with open('./PredSelfAtten/' + reason_method + '_' + 
                      bert_tmp + f'_{regresses_tmp}.pickle', 'wb') as f:
                pickle.dump({'bp':doc_ctb_segs, 'td':ques_ctb_segs, 
                             'bp_h':doc_h, 'td_h':ques_h}, f)
    # %%
    plt.close('all')
    regresses_tmp = 'VT_Resi'      # full, VT_Resi
    l_or_g = 0      # 0:local; 1:global
    bert_models = ['bert', 'albert', 'roberta']
    reason_methods = ['model_random', 'model_pre', 'model_fine'] 
    
    doc_ctb_berts = []
    ques_ctb_berts = []
    for bert_i, bert_tmp in enumerate(bert_models):
        doc_ctb_reas = []
        ques_ctb_reas = []
        for reason_method in reason_methods:
            with open('./PredSelfAtten/' + reason_method + '_' + 
                      bert_tmp + f'_{regresses_tmp}.pickle', 'rb') as f:
                bert_ctb = pickle.load(f)
            doc_ctb_segs_np = np.array(bert_ctb['bp'])[0]
            ques_ctb_segs_np = np.array(bert_ctb['td'])[0]
            
            doc_ctb_reas.append(doc_ctb_segs_np)
            ques_ctb_reas.append(ques_ctb_segs_np)
            
        doc_ctb_berts.append(doc_ctb_reas)
        ques_ctb_berts.append(ques_ctb_reas)
        
    # reason_method x g or l x 144
    doc_ctb_berts_np = np.array(doc_ctb_berts).mean(axis=0)[:, l_or_g, :]
    ques_ctb_berts_np = np.array(ques_ctb_berts).mean(axis=0)[:, l_or_g, :]
    
    
    doc_ctb_berts_layer_np = doc_ctb_berts_np.T.reshape((12, 12, -1), order='F')
    ques_ctb_berts_layer_np = ques_ctb_berts_np.T.reshape((12, 12, -1), order='F')
    
    fig_hand = plt.figure(figsize=(5, 1.8)) 
    axes_tmp = fig_hand.add_axes([0.1+0.4*0, 0.2, 0.38, 0.6])
    
    local_v = np.append(doc_ctb_berts_layer_np[:,:,:1], ques_ctb_berts_layer_np[:,:,:1], axis=-1)
    global_v = np.append(doc_ctb_berts_layer_np[:,:,1:], ques_ctb_berts_layer_np[:,:,1:], axis=-1)
    
    if l_or_g == 0:
        cmap_ = matplotlib.cm.get_cmap('Reds')
    else:
        cmap_ = matplotlib.cm.get_cmap('Blues')
    cmap_ = FunctionsZ.truncate_colormap(cmap_, minval=0.1, maxval=1.0, n=100)
    
    colors_l = cmap_(np.arange(3, 15)/15)
    DrawLine(
        doc_ctb_berts_layer_np, 
        cmap_, 
        axes_tmp
        )
    plt.ylim([0.07, 0.52])
    plt.yticks([0.1, 0.3, 0.5])
    
    axes_tmp = fig_hand.add_axes([0.1+0.45*1, 0.2, 0.4, 0.6])
    DrawLine(
        ques_ctb_berts_layer_np, 
        cmap_, 
        axes_tmp
        )
    plt.ylim([-0.01, 0.3])
    plt.yticks([0, 0.1, 0.2, 0.3])
    
            
            