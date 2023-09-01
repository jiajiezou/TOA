# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:41:10 2020

visualizing the results of eye measures prediction

@author: Jiajie
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neural_network import  MLPRegressor
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
import functools
from multiprocessing import Pool
from sklearn import preprocessing
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
import scipy.io
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from math import sqrt
import cmapy
from sklearn.model_selection import GridSearchCV
import string
from scipy import signal
import math
from tqdm import tqdm
    
def BertsPerm(
        pred_perm, reason_method, 
        regresses, exs, target_shows, bert_shows, types, fea_names
        ):
    perm_vs_regs = []
    for reg in regresses:
        perm_vs_exs = []
        for ex_tmp in exs:
            perm_vs_targets = []
            for target_show in target_shows:
                perm_vs_models = []
                for bert_show in bert_shows:
                    state = '_' + ex_tmp + '_' + reg 
                    perm_test = pred_perm.loc[
                        (pred_perm.State == state) & 
                        (pred_perm.ReasonMethod == reason_method) & 
                        (pred_perm.Model == bert_show) & 
                        (pred_perm.Target == target_show), :
                        ]
                    perm_v_l = [[
                        perm_test.loc[
                            (perm_test.Type==type_tmp) & 
                            (perm_test.Feature==fea_name), 
                            'Performance'].values 
                        for fea_name in fea_names] 
                        for type_tmp in types]
                    perm_vs_models.append(perm_v_l)
                perm_vs_targets.append(perm_vs_models)
            perm_vs_exs.append(perm_vs_targets)
        perm_vs_regs.append(perm_vs_exs)
    # regs x exs x target x model x types x feas
    perm_vs_np = np.array(perm_vs_regs)
    
    perm_bert = perm_vs_np[:, :, :, 0] 
    perm_albert = perm_vs_np[:, :, :, 1] 
    perm_roberta = perm_vs_np[:, :, :, 2] 
    perm_berts = np.append(perm_bert, perm_albert[:, :, :, :, -1:], axis=-2)
    perm_berts = np.append(perm_berts, perm_roberta[:, :, :, :, -1:], axis=-2)
    # regs x exs x target x types x feas
    return perm_berts

def cohen_d1(x,y):
    nx = len(x)
    ny = 1
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2) / dof)

def DrawBar(ys, ys_shuffle, bar_name, colors, group_name, ax):
    group_num = ys.shape[0]
    bar_num = ys.shape[1]
    
    bar_interval = 0.5
    x_interval = bar_num * bar_interval * 1.25
    xticks = (bar_num-1)*bar_interval/2 + np.arange(group_num)*x_interval
    
    p_v_feas = []
    for bar_i in range(bar_num):
        if len(ys.shape) > 2:
            y_tmp = ys[:, bar_i, 0]   
        else:
            y_tmp = ys[:, bar_i] 
            
        y_err_tmp = np.array(list(map(lambda x: np.nanstd(x) / math.sqrt(len(x)), y_tmp)))
        y_tmp = np.array(list(map(np.nanmean, y_tmp)))
        
        xs_ = np.arange(group_num)*x_interval+bar_i*bar_interval
        if colors.ndim==2:
            colors_ = colors[:, bar_i]
        else:    
            colors_ = colors[:, :, bar_i]
        
        for xs_i in range(len(xs_)):
            plt.bar(
                xs_[xs_i], 
                height=y_tmp[xs_i], 
                width=bar_interval*0.8, 
                color=colors_[xs_i],  
                label=bar_name[bar_i],
                ) 
            
            plotline, caps, barlinecols = plt.errorbar(
                xs_[xs_i], 
                y_tmp[xs_i], yerr=y_err_tmp[xs_i], linestyle='',
                ecolor=colors_[xs_i],
                capsize=2, 
                capthick=0.8,
                ) 
            plt.setp(barlinecols[0], 
                      color=colors_[xs_i], 
                     linewidth=0.8)
        
        y_shuff_tmp = ys_shuffle[:, bar_i, :]
        y_shuff_st_tmp = np.sort(y_shuff_tmp, axis=-1)
        
        p_v_exs = []
        for p_i in range(group_num):
            
            if y_tmp[p_i] < 0.3:
                plt.text(
                    p_i*x_interval+bar_i*bar_interval+0.02/2.54, 
                    # y_tmp[p_i] + 0.04, bar_name[bar_i], 
                    y_tmp[p_i] + y_err_tmp[p_i] + 0.05, bar_name[bar_i], 
                    fontsize=8,  
                    # fontsize=4,  
                    ha='center', va='bottom', 
                    color='k', rotation=90
                    ) 
            else:
                plt.text(
                    p_i*x_interval+bar_i*bar_interval+0.02/2.54, 
                    # 0.04, bar_name[bar_i], 
                    0.05, bar_name[bar_i], 
                    fontsize=8, 
                    # fontsize=4, 
                    ha='center', va='bottom', 
                    color='w', rotation=90
                    ) 
                    
            # p_value = (sum(y_shuff_st_tmp[p_i]>y_tmp[p_i])+1) / (len(y_shuff_st_tmp[p_i])+1)
            p_value = (sum(y_shuff_st_tmp[-1]>y_tmp[p_i])+1) / (len(y_shuff_st_tmp[-1])+1)
            
            
            if (p_value < 0.05) and (p_value > 0.01):
                plt.text(
                    p_i*x_interval+bar_i*bar_interval+0.02/2.54, 
                    y_tmp[p_i] + y_err_tmp[p_i], '*', 
                    fontsize=9, 
                    # fontsize=7, 
                    ha='center', va='center', 
                    color=colors_[p_i]
                    ) 
            elif (p_value < 0.01) and (p_value > 0.001):
                plt.text(
                    p_i*x_interval+bar_i*bar_interval+0.02/2.54, 
                    y_tmp[p_i] + y_err_tmp[p_i], '**', 
                    fontsize=9, 
                    # fontsize=7, 
                    ha='center', va='center', 
                    color=colors_[p_i]
                    ) 
            elif p_value < 0.001:
                plt.text(
                    p_i*x_interval+bar_i*bar_interval+0.02/2.54, 
                    y_tmp[p_i] + y_err_tmp[p_i], '***', 
                    fontsize=9, 
                    # fontsize=7, 
                    ha='center', va='center',
                    color=colors_[p_i]
                    ) 
                plt.pause(0.2)
            p_v_exs.append(p_value)
        p_v_feas.append(p_v_exs)
        
    # plt.legend(ncol=1)
    plt.xticks(xticks, group_name)
    plt.xlim(xticks[0]-bar_num/2*bar_interval-bar_interval*0.5, xticks[-1]+bar_num/2*bar_interval+bar_interval*0.5)
    
    # plt.xlim(xticks[0]-xticks[0]*2, xticks[-1]+xticks[0]*2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.pause(1)
    return np.array(p_v_feas), xticks


def Str2Num_flat(str_array):
    # str_array: 2d
    ys_num = []
    for idx1, str_tmp1 in enumerate(str_array):
        ys_num1 = []
        for idx2, str_tmp2 in enumerate(str_tmp1):
            ys_tmp_num = np.array(str_tmp2[0].split(', ')).astype(float)
            ys_num1.append(ys_tmp_num)
        ys_num.append(ys_num1)
    return np.array(ys_num)

def MeaSigTest(xs, times):
    ps = []
    for idx0 in range(xs.shape[0]):
        for idx1 in range(idx0+1, xs.shape[0]):
            diff_boot = FunctionsZ.Bootstrap(
                times, np.mean, xs[idx1]-xs[idx0]
                )
            p = 2*min(sum(diff_boot < 0)+1, 
                          sum(diff_boot > 0)+1)/(boot_times+1)
            ps.append(p)
    return ps

def BarEye_types(TestCoef, TestCoef_shuffle, xlabels, colors_f, legends, ax):

    TestCoef = np.ma.masked_array(TestCoef, np.isnan(TestCoef))
    coef_exs = TestCoef[0]
    # coef_exs: experiments x features x 1
    
    # chance level
    TestCoef_shuffle = np.ma.masked_array(TestCoef_shuffle, np.isnan(TestCoef_shuffle))
    coef_exs_shuffle = TestCoef_shuffle[0]
    
    plt.plot([-1, 50], [0, 0], '-', color=[0, 0, 0], linewidth=0.5, alpha=0.1)
    
    
    # coef_exs: exs x features
    coef_exs_ = np.swapaxes(coef_exs, 0, 1)
    coef_exs_[-3] = coef_exs_[-3:].mean(axis=0)
    coef_exs_ = coef_exs_[:-2]
    
    coef_exs_shuffle_ = np.swapaxes(coef_exs_shuffle, 0, 1)
    coef_exs_shuffle_[-3] = coef_exs_shuffle_[-3:].mean(axis=0)
    coef_exs_shuffle_ = coef_exs_shuffle_[:-2]
    
    # DNN remove
    coef_exs_ = coef_exs_[:-1]
    # coef_exs_shuffle_ = coef_exs_shuffle_[:-1]
    
    p_values = DrawBar(
        coef_exs_, coef_exs_shuffle_, 
        legends, colors_f, xlabels, ax
        )
    plt.ylim(-0.1, 0.8)
    plt.yticks([0, 0.4, 0.8])
    return np.array(p_values)

#%%Classification
if __name__ == '__main__':
    FunctionsZ.FontSetting()
    regresses = ['full', 'V_Resi', 'VT_Resi', 'VTR_Resi']
    exs = ['800', 'MixedGoalL2', 'MixedNative', 'Mixed']
    states = [
        '_' + ex_tmp + '_' + regresses_tmp 
        for ex_tmp in exs for regresses_tmp in regresses
        ]
    
    # %%
    cmap = matplotlib.cm.get_cmap('Reds')
    colors_f = [
        [1.0000, 0.1429, 0], [1.0000, 0.4603, 0], [1.0000, 0.7778, 0], 
        [0, 0.1429, 0.9286], [0, 0.4603, 0.7698], [0, 0.7778, 0.6111]
        ]
    pred_perm = pd.read_csv('./PredEyeCSV/EyePred.csv', index_col=0)
    pred_perm_shuffle = pd.read_csv('./PredEyeCSV/EyePred_chance.csv')
    
    types = ['LPurpose', 'Fact', 'Limply', 'Mainly', 'Title', 'GPurpose']
    
    fea_names = ['visual_layout', 'text_smp', 'QuesReFea_name', 'EditMean', 'GloveMean', 'SAR_Atten', 'AttenFea']
    xlabels = ['layout', 'word', 'relevance', 'orthographic', 'semantic', 'SAR', 'BERT', 'ALBERT', 'RoBERTa']
    
    bert_shows = ['bert', 'albert', 'roberta']
    target_shows = [
        'IA_DWELL_TIME_lp', 
        'IA_FIRST_RUN_DWELL_TIME_lp', 
        'IA_RUN_COUNT_Del0_lp'
        ]
    reason_methods = ['model_random', 'model_pre', 'model_fine']
    # merged Berts of different reason_method
    perm_berts = None
    for reason_method in tqdm(reason_methods):
        # perm_bert_tmp: regs x exs x target x types x feas x shuffle_num
        perm_bert_tmp = BertsPerm(
            pred_perm, reason_method, 
            regresses, exs, target_shows, bert_shows, types, fea_names
            )    
        # Todo: chance level
        # regs x exs x target x types x feas x shuffle_num
        perm_bert_shuffle_tmp = BertsPerm(
            pred_perm_shuffle, 'CLS', 
            ['full']*len(regresses), exs, ['IA_DWELL_TIME_lp']*len(target_shows), 
            bert_shows, types, ['AttenFea']*len(fea_names)
            )    
        if perm_berts is None:
            perm_berts = copy.deepcopy(perm_bert_tmp)
            perm_berts_shuffle = copy.deepcopy(perm_bert_shuffle_tmp)
        else:
            perm_berts = np.append(perm_berts, perm_bert_tmp[:, :, :, :, -3:], axis=4)
            perm_berts_shuffle = np.append(perm_berts_shuffle, perm_bert_shuffle_tmp[:, :, :, :, -3:], axis=4)
            
    # perm_bert_tmp: regs x exs x target x types x feas(6+3*5) x shuffle_num
    perm_berts_ = copy.deepcopy(perm_berts)
    perm_berts_shuffle_ = copy.deepcopy(perm_berts_shuffle)
    
    # V_Resi, word -> full, word
    perm_berts_[0, :, :, :, 1] = copy.deepcopy(perm_berts_[1, :, :, :, 1])
    # VT_Resi, r -> full, r
    perm_berts_[0, :, :, :, 2] = copy.deepcopy(perm_berts_[2, :, :, :, 2])
    
    # ----------draw figures
    # boot_times = 50000
    boot_times = 50
    
    # %%
    # For Ex 1: cmp PP for different models
    fea_type = 'DNN'    # DNN, hand_crafted, All
    full_or_resi = 0    # 'full', 'VT_Resi', 'VTR_Resi'
    
    p_values_dict = {}
    colors_f = [
        [1.0000, 0.1429, 0], [1.0000, 0.4603, 0], [1.0000, 0.7778, 0], 
        [0, 0.1429, 0.9286], [0, 0.4603, 0.7698], [0, 0.7778, 0.6111]
        ]
    
    # real
    perm_berts_1 = perm_berts_[full_or_resi, 0, 0, :, :]
    perm_berts_num1 = Str2Num_flat(perm_berts_1)
    
    if 'DNN' in fea_type:
        # merge models
        perm_berts_num1_ = []
        for perm_berts_tmp in perm_berts_num1:
            perm_berts_tmp_arr = np.array(list(perm_berts_tmp))     # features x docs
            # difference across DNN
            doc_num = np.array(list(map(np.shape, perm_berts_tmp_arr)))
            unmatch_idxs = np.where(doc_num != doc_num[0])[0]
            for unmatch_idx in unmatch_idxs:
                if unmatch_idx in [6, 7, 8]:
                    perm_berts_tmp_arr[unmatch_idx] = perm_berts_tmp_arr[6]
                    perm_berts_tmp_arr = np.array(list(perm_berts_tmp_arr))
                elif unmatch_idx in [9, 10, 11]:
                    perm_berts_tmp_arr[unmatch_idx] = perm_berts_tmp_arr[9]
                    perm_berts_tmp_arr = np.array(list(perm_berts_tmp_arr))
                elif unmatch_idx in [12, 13, 14]:
                    perm_berts_tmp_arr[unmatch_idx] = perm_berts_tmp_arr[12]
                    perm_berts_tmp_arr = np.array(list(perm_berts_tmp_arr))
                else:
                    assert 1 == 0
            # mean across DNN
            perm_berts_tmp_arr_ = copy.deepcopy(perm_berts_tmp_arr)
            # fine-tuned
            perm_berts_tmp_arr_[-3, :] = perm_berts_tmp_arr[-3:, :].mean(axis=0)
            # pretrained
            perm_berts_tmp_arr_[-6, :] = perm_berts_tmp_arr[-6:-3, :].mean(axis=0)
            # random
            perm_berts_tmp_arr_[-9, :] = perm_berts_tmp_arr[-9:-6, :].mean(axis=0)
            
            perm_berts_tmp_arr_ = np.append(
                np.append(perm_berts_tmp_arr_[:-8, :], 
                          perm_berts_tmp_arr_[-6:-5, :], axis=0), 
                perm_berts_tmp_arr_[-3:-2, :], axis=0 
                )
            
            perm_berts_num1_.append(list(perm_berts_tmp_arr_))
        perm_berts_num1 = np.array(list(perm_berts_num1_))
        
    # chance level
    perm_berts_shuffle_1 = copy.deepcopy(perm_berts_shuffle_)
    perm_berts_shuffle_1 = perm_berts_shuffle_1[full_or_resi, 0, 0, :, :]
    
    if 'DNN' in fea_type:
        # merge models
        perm_berts_shuffle_1[:, -3] = perm_berts_shuffle_1[:, -3:].mean(axis=1)
        perm_berts_shuffle_1[:, -6] = perm_berts_shuffle_1[:, -6:-3].mean(axis=1)
        perm_berts_shuffle_1[:, -9] = perm_berts_shuffle_1[:, -9:-6].mean(axis=1)
        perm_berts_shuffle_1 = np.append(
            np.append(perm_berts_shuffle_1[:, :-8], 
                      perm_berts_shuffle_1[:, -6:-5], axis=1),
            perm_berts_shuffle_1[:, -3:-2], axis=1
            )
        
    if fea_type == 'DNN':
        # ------DNN models
        type_legend = ['SAR', 'trans_rand', 'trans_pre', 'trans_fine']
        perm_berts_1 = perm_berts_num1[:, -len(type_legend):]
        perm_berts_shuffle_1 = perm_berts_shuffle_1[:, -len(type_legend):]
        fig_hand, axes = plt.subplots(1, 1, figsize=(5.45, 2.2))
    elif fea_type == 'hand_crafted':
        # ------hand-crafted features
        type_legend = ['layout', 'word', 'relevance']
        perm_berts_1 = perm_berts_num1[:, :len(type_legend)]
        perm_berts_shuffle_1 = perm_berts_shuffle_1[:, :len(type_legend)]
        fig_hand, axes = plt.subplots(1, 1, figsize=(5.45, 2.2))
    elif fea_type == 'All':
        # ------all features
        type_legend = [
            'orthographic', 'semantic', 'SAR', 
            'BERT_rand', 'ALBERT_rand', 'RoBERTa_rand', 
            'BERT_pre', 'ALBERT_pre', 'RoBERTa_pre', 
            'BERT_fine', 'ALBERT_fine', 'RoBERTa_fine'
            ]
        
        perm_berts_1 = perm_berts_num1[:, 3:]
        perm_berts_shuffle_1 = perm_berts_shuffle_1[:, 3:]
        fig_hand, axes = plt.subplots(1, 1, figsize=(8.5, 2.2))
        
    plt.sca(axes)
    plt.plot([-1, 50], [0, 0], '-', color=[0, 0, 0], linewidth=0.5, alpha=0.1)
    
    xlabels_1 = ['Cause', 'Fact', 'Inference', 'Theme', 'Title', 'Purpose']
    
    # group_num * bar_num in each group
    colors_f_ = np.tile(np.array(colors_f)[:,:,np.newaxis], (1, 1, perm_berts_1.shape[-1]))
    p_values, _ = DrawBar(
        perm_berts_1, perm_berts_shuffle_1, 
        type_legend, colors_f_, xlabels_1, axes
        )
    # p_values: type x features
    plt.ylim(-0.05, 0.7)
    plt.yticks([0, 0.2, 0.4, 0.6])
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.24, wspace=0.1, hspace=0.3) 
    plt.show() 
    
    # significance test: local VS global
    l_g_p = []
    for perm_berts_fea in perm_berts_1.T:
        perm_berts_l = np.concatenate(perm_berts_fea[:3])
        perm_berts_g = np.concatenate(perm_berts_fea[3:])
        
        perm_l_boot = FunctionsZ.Bootstrap(boot_times, np.nanmean, perm_berts_l)
        
        p_tmp = 2*min(sum(perm_l_boot<np.nanmean(perm_berts_g))+1, 
                      sum(perm_l_boot>np.nanmean(perm_berts_g))+1)/(boot_times+1)
        l_g_p.append(p_tmp)
    
    # significance test: features
    feas_p = []
    for idx0 in range(perm_berts_1.shape[1]):
        for idx1 in range(idx0+1, perm_berts_1.shape[1]):
            feas_type = []
            for type_i in range(perm_berts_1.shape[0]):
                perm_fea_0 = perm_berts_1[type_i, idx0]
                perm_fea_1 = perm_berts_1[type_i, idx1]
                
                perm_fea_0_boot = FunctionsZ.Bootstrap(boot_times, np.nanmean, perm_fea_0)
                p_tmp = 2*min(sum(perm_fea_0_boot<np.nanmean(perm_fea_1))+1, 
                              sum(perm_fea_0_boot>np.nanmean(perm_fea_1))+1)/(boot_times+1)
                feas_type.append(p_tmp)
            feas_p.append(feas_type)
    feas_p = np.array(feas_p)
    
    
    # fdr correction
    # p_values, l_g_p, feas_p
    p_values_flat = p_values.reshape((-1, ), order='F')
    p_values_fdr = FunctionsZ.fdr_bh(p_values_flat).reshape(p_values.shape, order='F') 
    
    feas_p_flat = feas_p.reshape((-1, ), order='F')
    feas_p_fdr = FunctionsZ.fdr_bh(feas_p_flat).reshape(feas_p.shape, order='F')
    
    l_g_p_fdr = FunctionsZ.fdr_bh(l_g_p)
    # write to file or print
    p_excel = pd.DataFrame(columns = ['row_name'] + xlabels_1 + ['L vs G'])
    if fea_type == 'DNN':
        p_excel['row_name'] = type_legend + ['S vs R', 'S vs P', 'S vs F', 'R vs P', 'R vs F', 'P vs F']
        p_excel.iloc[:4, 1:-1] = p_values_fdr
        p_excel.iloc[4:, 1:-1] = feas_p_fdr
        p_excel.iloc[:4, -1] = l_g_p_fdr
    elif fea_type == 'hand_crafted':
        p_excel['row_name'] = type_legend + ['layout vs word', 'layout vs relev', 'word vs relev']
        p_excel.iloc[:3, 1:-1] = p_values_fdr
        p_excel.iloc[3:, 1:-1] = feas_p_fdr
        p_excel.iloc[:3, -1] = l_g_p_fdr
    print(p_excel)
    # p_excel.to_excel(yourpath/Pred_reading_time_ex1_{fea_type}_{full_or_resi}.xls')
    
    # %%
    # cmp early and late measures
    # regs x exs x target x types x feas x shuffle_num
    ex_test = 0     # 0~3: ex 1~4
    
    type_legend = ['Cause', 'Fact', 'Inference', 'Theme', 'Title', 'Purpose']
    target_shows_smp = ['GD', 'RC']
    x_label = ['position', 'word', 'relevance']
        
    # meas x types
    for l_or_g in [0, 1]:
        if l_or_g == 0:
            cmap_ = matplotlib.cm.get_cmap('Reds')
            fig_hand = plt.figure(figsize=(5, 1.8))
        else:
            cmap_ = matplotlib.cm.get_cmap('Blues')
            fig_hand = plt.figure(figsize=(5, 1.8))
        colors_f = [cmap_(0.5), cmap_(0.9)]
        
        p_chance_meas = []
        mea_p_meas = []
        for axes_i in range(perm_berts_.shape[-2]-3-3*3):
            # bar plot
            axes_tmp = fig_hand.add_axes([0.1+0.38/3*axes_i, 0.2, 0.38/3, 0.6])
            
            plt.sca(axes_tmp)
            perm_berts_mea_type = perm_berts_[0, ex_test, 1:, :, axes_i]
            perm_berts_mea_type_shuffle = perm_berts_shuffle_[0, ex_test, 1:, :, axes_i]
            plt.plot([-1, 50], [0, 0], '-', color=[0, 0, 0], linewidth=0.5, alpha=0.1)
            
            perm_berts_mea_type_num = Str2Num_flat(perm_berts_mea_type)
            
            # mean across subtypes
            perm_berts_mea_type_num_ = []
            for berts_mea_tmp in perm_berts_mea_type_num:
                berts_mea_tmp_l = np.concatenate(berts_mea_tmp[:3])
                berts_mea_tmp_g = np.concatenate(berts_mea_tmp[3:])
                perm_berts_mea_type_num_.append([berts_mea_tmp_l, berts_mea_tmp_g])
                
            perm_berts_mea_type_num_ = np.array(perm_berts_mea_type_num_) 
            perm_berts_mea_type_num = perm_berts_mea_type_num_ # meas x types
            
            perm_berts_mea_l_or_g = perm_berts_mea_type_num[:, l_or_g:l_or_g+1]
            perm_berts_mea_l_or_g_shuffle = perm_berts_mea_type_shuffle[:, l_or_g:l_or_g+1]
            
            colors_f_ = np.tile(np.array(colors_f)[:,:, np.newaxis], (1, 1, perm_berts_mea_l_or_g.shape[-1]))
            # plot
            p_values, xticks = DrawBar(
                perm_berts_mea_l_or_g, perm_berts_mea_l_or_g_shuffle, 
                type_legend, colors_f_, target_shows_smp, axes_tmp
                )
            p_chance_meas.append(p_values)
            
            # p value for early vs late
            mea_p = MeaSigTest(perm_berts_mea_type_num[:, l_or_g], boot_times)
            mea_p_meas.append(mea_p)
            
            plt.ylim(-0.02, 0.33)
            plt.yticks([0, 0.1, 0.2, 0.3])
            axes_tmp.spines['bottom'].set_visible(True)
    
            if axes_i > 0:
                axes_tmp.get_yaxis().set_visible(False)
                axes_tmp.spines['left'].set_visible(False)
            
        # fdr correction
        p_chance_meas_np = np.array(p_chance_meas)[:, 0, :]
        p_values_flat = p_chance_meas_np.reshape((-1, ), order='F')
        p_values_fdr = FunctionsZ.fdr_bh(p_values_flat).reshape(p_chance_meas_np.shape, order='F')
        
        mea_p_meas_np = np.array(mea_p_meas)[:, 0]
        p_mea_diff_fdr = FunctionsZ.fdr_bh(mea_p_meas_np)
        
        p_excel = pd.DataFrame(columns = ['row_name'] + x_label)
        p_excel['row_name'] = ['GD', 'RC'] + ['GD vs RC']
        p_excel.iloc[:-1, 1:] = p_values_fdr.T
        p_excel.iloc[-1:, 1:] = p_mea_diff_fdr
        print(p_excel)
        # p_excel.to_excel(yourpath/Pred_early_late_{ex_test}_{l_or_g}.xlsx')
        
    # %%
    # cmp exs
    # regs x exs x target x types x feas x shuffle_num
    l_or_g = 'local'    # local, global
    fea_type = 'DNN'    # hand_crafted, DNN
    target_i = 0    # 0~2: dwell time, gaze duration, and rereading counts
    
    p_values_dict = {}
    type_legend = ['Ex 1', 'Ex 2', 'Ex 3', 'Ex 4']
    colors_f = [
        '#E76F51', '#F4A261', '#E9C46A', '#6BA292', 
        ]
    
    if l_or_g == 'local':
        perm_berts_l = perm_berts_[0, :, target_i, :3] # local
    else:
        perm_berts_l = perm_berts_[0, :, target_i, 3:] # global
            
    perm_berts_l_ = np.swapaxes(perm_berts_l, 1, 2)
    perm_berts_l_join1 = []
    for tmp1 in perm_berts_l_:
        perm_berts_l_join2 = []
        for tmp2 in tmp1:
            tmp2_join = ', '.join(np.concatenate(tmp2))
            perm_berts_l_join2.append([tmp2_join])
        perm_berts_l_join1.append(perm_berts_l_join2)
    perm_berts_l = np.array(perm_berts_l_join1)
    # to number
    perm_berts_l_num = Str2Num_flat(perm_berts_l)
    
    # merge models
    perm_berts_l_num_ = []
    for perm_berts_tmp in perm_berts_l_num:
        perm_berts_tmp_arr = np.array(list(perm_berts_tmp))
        model_idx = list(np.arange(6)) + list(np.arange(9, 15))
        perm_berts_tmp_arr = perm_berts_tmp_arr[model_idx]
        # difference across DNN
        doc_num = np.array(list(map(np.shape, perm_berts_tmp_arr)))
        unmatch_idxs = np.where(doc_num != doc_num[0])[0]
        for unmatch_idx in unmatch_idxs:
            if unmatch_idx in [6, 7, 8]:
                perm_berts_tmp_arr[unmatch_idx] = perm_berts_tmp_arr[6]
                perm_berts_tmp_arr = np.array(list(perm_berts_tmp_arr))
            elif unmatch_idx in [9, 10, 11]:
                perm_berts_tmp_arr[unmatch_idx] = perm_berts_tmp_arr[9]
                perm_berts_tmp_arr = np.array(list(perm_berts_tmp_arr))
            else:
                assert 1 == 0
                
        # mean across DNN
        perm_berts_tmp_arr_ = copy.deepcopy(perm_berts_tmp_arr)
        # fine-tuned
        perm_berts_tmp_arr_[-3, :] = perm_berts_tmp_arr[-3:, :].mean(axis=0)
        # pretrained
        perm_berts_tmp_arr_[-6, :] = perm_berts_tmp_arr[-6:-3, :].mean(axis=0)
        perm_berts_tmp_arr_ = np.append(perm_berts_tmp_arr_[:-5, :], perm_berts_tmp_arr_[-3:-2, :], axis=0)
        perm_berts_l_num_.append(list(perm_berts_tmp_arr_))
    perm_berts_l_num = np.array(list(perm_berts_l_num_))
    
    if l_or_g == 'local':
        perm_berts_shuffle_l = perm_berts_shuffle_[0, :, 0, :3].mean(axis=1) # local
    else:
        perm_berts_shuffle_l = perm_berts_shuffle_[0, :, 0, 3:].mean(axis=1) # global
        
    # merge models
    perm_berts_shuffle_l[:, -3] = perm_berts_shuffle_l[:, -3:].mean(axis=1)
    perm_berts_shuffle_l[:, -6] = perm_berts_shuffle_l[:, -6:-3].mean(axis=1)
    perm_berts_shuffle_l = np.append(perm_berts_shuffle_l[:, :-5], perm_berts_shuffle_l[:, -3:-2], axis=1)
    
    if fea_type == 'DNN':
        # ------all features
        xlabels_1 = [
            'transformer$\mathregular{_p}$', 'tramsformer$\mathregular{_f}$', 'difference'
            ]
        perm_berts_tmp = np.swapaxes(perm_berts_l_num[:, 6:], 0, 1)
        perm_berts_shuffle_tmp = np.swapaxes(perm_berts_shuffle_l[:, 6:], 0, 1)
        
        perm_berts_tmp = np.append(perm_berts_tmp, perm_berts_tmp[-1:]-perm_berts_tmp[-2:-1], axis=0)
        # perm_berts_shuffle_tmp = np.append(perm_berts_shuffle_tmp, perm_berts_shuffle_tmp[-1:]-perm_berts_shuffle_tmp[-2:-1], axis=0)
        perm_berts_shuffle_tmp = np.append(perm_berts_shuffle_tmp, perm_berts_shuffle_tmp[-1:], axis=0)
    elif fea_type == 'hand_crafted':
        # ------hand-crafted features
        xlabels_1 = ['layout', 'word', 'relevance']
        perm_berts_tmp = np.swapaxes(perm_berts_l_num[:, :3], 0, 1)
        perm_berts_shuffle_tmp = np.swapaxes(perm_berts_shuffle_l[:, :3], 0, 1)
    
    fig_hand, axes = plt.subplots(1, 1, figsize=(3.5, 1.8))
    plt.sca(axes)
    plt.plot([-1, 50], [0, 0], '-', color=[0, 0, 0], linewidth=0.5, alpha=0.1)
    
    colors_f_ = np.tile(np.array(colors_f)[np.newaxis,:], (perm_berts_tmp.shape[-1], 1))
    
    p_values, _ = DrawBar(perm_berts_tmp, perm_berts_shuffle_tmp, 
                       type_legend, colors_f_, xlabels_1, axes)
    # pvalues: exs x features
    plt.ylim(-0.1, 0.8)
    plt.yticks([0, 0.4, 0.8])
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.24, wspace=0.1, hspace=0.3)
    
    
    # significance test: exs
    # feas_exs_p: features x exs_cmp
    feas_exs_p = []
    for perm_berts_tmp_1 in perm_berts_tmp:
        exs_p = []
        for idx0 in range(perm_berts_tmp_1.shape[0]):
            for idx1 in range(idx0+1, perm_berts_tmp_1.shape[0]):
                perm_fea_0 = perm_berts_tmp_1[idx0]
                perm_fea_1 = perm_berts_tmp_1[idx1]
                
                perm_fea_0_boot = FunctionsZ.Bootstrap(boot_times, np.nanmean, perm_fea_0)
                p_tmp = 2*min(sum(perm_fea_0_boot<np.nanmean(perm_fea_1))+1, 
                              sum(perm_fea_0_boot>np.nanmean(perm_fea_1))+1)/(boot_times+1)
                exs_p.append(p_tmp)
        feas_exs_p.append(exs_p)
    
    p_values_flat = p_values.reshape((-1, ), order='F')
    feas_exs_p_np = np.array(feas_exs_p)
    feas_exs_p_flat = feas_exs_p_np.reshape((-1, ), order='F')
    
    # fdr correction
    ps = np.concatenate((p_values_flat, feas_exs_p_flat), axis=0)
    ps_fdr = FunctionsZ.fdr_bh(ps)
    p_values_fdr = ps_fdr[:len(p_values_flat)].reshape(p_values.shape, order='F')
    feas_exs_p_fdr = ps_fdr[len(p_values_flat):].reshape(feas_exs_p_np.shape, order='F')
    
    # save to excel: cmp to chance level
    p_excel = pd.DataFrame(columns = ['row_name'] + ['Ex 1', 'Ex 2', 'Ex 3', 'Ex 4'])
    p_excel['row_name'] = xlabels_1
    p_excel.iloc[:3, 1:] = p_values_fdr.T[:, :4]
    # p_excel.to_excel(yourpath/exs_{fea_type}_{l_or_g}.xls)
    print(p_excel)
    
    
    print('='*100)
    # save to excel: cmp between experiments
    p_excel = pd.DataFrame(
        columns = ['row_name'] + ['Ex 1 vs Ex 2', 'Ex 1 vs Ex 3', 
                                  'Ex 1 vs Ex 4', 'Ex 2 vs Ex 3', 
                                  'Ex 2 vs Ex 4', 'Ex 3 vs Ex 4', ]
        )
    p_excel['row_name'] = xlabels_1
    p_excel.iloc[:, 1:] = feas_exs_p_fdr    
    # p_excel.to_excel(yourpath/exs_cmp_{fea_type}_{l_or_g}.xls')
    print(p_excel)
    
    
    