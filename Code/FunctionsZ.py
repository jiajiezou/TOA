# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:02:20 2020

Functions

@author: Jiajie
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.utils import resample
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score
from scipy import stats
from scipy.stats import norm
import math
import string
import pandas as pd
import copy
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import json
from matplotlib.colors import LinearSegmentedColormap
from difflib import SequenceMatcher
from statsmodels.stats.multitest import fdrcorrection as fdr_cor

# %% modeling
def Prepocessing_func(Feas_train, Feas_test, Labels_train, Labels_test
                      , varian_ratio_tol=1, dele_pca_cpn=None):
    """
    Data preprocessing: PCA, train set and test set split, and normalization
    """   
    #Normalizing
    scaler_fea1 = preprocessing.StandardScaler().fit(Feas_train)
    Feas_train = scaler_fea1.transform(Feas_train)
    Feas_test = scaler_fea1.transform(Feas_test)
    Reducter = None
    #keep the first n components
    if (varian_ratio_tol < 1) and (Feas_train.shape[-1] > 50):
        #Reducter = PCA()
        Reducter = IncrementalPCA()
        Reducter.fit(Feas_train)
        Feas_train_redu = Reducter.transform(Feas_train)
        Feas_test_redu = Reducter.transform(Feas_test)
        variance_ratio = np.cumsum(Reducter.explained_variance_ratio_)
        cpn_tol_flag = variance_ratio < varian_ratio_tol
        Feas_train = Feas_train_redu[:, cpn_tol_flag]
        Feas_test = Feas_test_redu[:, cpn_tol_flag]
    elif varian_ratio_tol > 1:
        Reducter = LinearDiscriminantAnalysis()
        Reducter.fit(Feas_train, Labels_train)
        Feas_train_redu = Reducter.transform(Feas_train)
        Feas_test_redu = Reducter.transform(Feas_test)
        Feas_train = Feas_train_redu[:, :varian_ratio_tol]
        Feas_test = Feas_test_redu[:, :varian_ratio_tol]
        
    #delete the first n components
    if dele_pca_cpn!=None:
        Reducter = PCA()
        Reducter.fit(Feas_train)
        Feas_train_redu = Reducter.transform(Feas_train)
        Feas_test_redu = Reducter.transform(Feas_test)
        Feas_train = Feas_train_redu[:, dele_pca_cpn:]
        Feas_test = Feas_test_redu[:, dele_pca_cpn:]
        
    #Normalizing
    scaler_fea2 = preprocessing.StandardScaler().fit(Feas_train)
    Feas_train = scaler_fea2.transform(Feas_train)
    Feas_test = scaler_fea2.transform(Feas_test)
    return (Feas_train, Feas_test, Labels_train, Labels_test, 
            scaler_fea1, Reducter, scaler_fea2)

def regressor_comparison_func(Feas_train, Feas_test, Labels_train, Labels_test, Regressor):
    model_comparison = pd.DataFrame(
        columns=['Regressor', 'TrainError', 'TrainCoef', 'TrainEVS', 'TrainR2', 'TrainPE', 
                 'TestError', 'TestCoef', 'TestEVS', 'TestR2', 'TestPE', 'TestPred', 'y_test']
        )
    X_train = Feas_train
    y_train = Labels_train
    X_test = Feas_test
    y_test = Labels_test
    for i, rgs in enumerate(Regressor):
        #Modeling
        rgs.fit(X_train, y_train)
        #train error
        TrainPred = rgs.predict(X_train)
        TrainError = mean_squared_error(y_train, TrainPred)
        TrainCoef, _ = stats.pearsonr(y_train, TrainPred)
        TrainEVS = explained_variance_score(y_train, TrainPred)
        TrainR2 = r2_score(y_train, TrainPred)
        TrainPE = mean_absolute_percentage_error(y_train, TrainPred)
        
        #test accuracy
        TestPred = rgs.predict(X_test)
        TestError = mean_squared_error(y_test, TestPred)
        TestCoef, _ = stats.pearsonr(y_test, TestPred)
        TestEVS = explained_variance_score(y_test, TestPred)
        TestR2 = r2_score(y_test, TestPred)
        TestPE = mean_absolute_percentage_error(y_test, TestPred)
        
        model_comparison.loc[len(model_comparison.index), :] = \
        [rgs, TrainError, TrainCoef, TrainEVS, TrainR2, TrainPE, TestError, 
         TestCoef, TestEVS, TestR2, TestPE, TestPred[:100], y_test[:100]]
        
    return model_comparison

def regressor_residual_func(
        X_train_out, X_test_out, X_train, X_test, 
        y_train, y_test, Regressor
        ):
    model_comparison = pd.DataFrame(
        columns=['RegressorOut', 'RegressorResi', 
                 'TrainCoefOut', 'TestCoefOut', 'TrainPE_Out', 'TestPE_Out', 
                 'TrainCoefResi', 'TestCoefResi', 'TrainPE_Resi', 'TestPE_Resi', 
                 'y_test', 'y_test_resi', 'y_test_pred_resi']
        )
    
    # ----------regress out---------- #
    rgs_out = Regressor[0]
    rgs_out.fit(X_train_out, y_train)
    # train
    y_train_pred_out = rgs_out.predict(X_train_out)
    train_coef_out, _ = stats.pearsonr(y_train, y_train_pred_out)
    TrainPE_out = mean_absolute_percentage_error(y_train, y_train_pred_out)
    # test
    y_test_pred_out = rgs_out.predict(X_test_out)
    test_coef_out, _ = stats.pearsonr(y_test, y_test_pred_out)
    TestPE_out = mean_absolute_percentage_error(y_test, y_test_pred_out)
    
    # ----------residual error---------- #
    y_train_resi = y_train - y_train_pred_out
    y_test_resi = y_test - y_test_pred_out
    
    rgs_resi = Regressor[1]
    rgs_resi.fit(X_train, y_train_resi)
    # train
    y_train_pred_resi = rgs_resi.predict(X_train)
    train_coef_resi, _ = stats.pearsonr(y_train_resi, y_train_pred_resi)
    TrainPE_resi = mean_absolute_percentage_error(y_train_resi, y_train_pred_resi)
    
    # test
    y_test_pred_resi = rgs_resi.predict(X_test)
    test_coef_resi, _ = stats.pearsonr(y_test_resi, y_test_pred_resi)
    TestPE_resi = mean_absolute_percentage_error(y_test_resi, y_test_pred_resi)
    
    model_comparison.loc[len(model_comparison.index), :] = [
        rgs_out, rgs_resi, 
        train_coef_out, test_coef_out, TrainPE_out, TestPE_out, 
        train_coef_resi, test_coef_resi, TrainPE_resi, TestPE_resi, 
        y_test[:100], y_test_resi[:100], y_test_pred_resi[:100]
        ]
        
    return model_comparison

# %% plotting
def FontSetting():
    plt.rcParams.update(
        {'font.size': 10,
          'font.family': ['Arial'],
          'font.weight': 'normal',
          # 'legend.fontsize': 'x-small',
          'legend.fontsize': 'medium',
          'axes.labelsize': 'Large',
          'axes.titlesize': 'Large',
          'axes.titleweight': 'normal',
          })
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# %% statistical analysis
def Bootstrap(times, func, data):
    all_results = []
    for i in range(times):
        permutated = np.random.choice(data, size=len(data), replace=True)
        permut_result = func(permutated)
        all_results.append(permut_result)
    return np.array(all_results)
        
def fdr_bh(vector):
    return fdr_cor(vector)[1]