# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:41:10 2020

predict self attention

@author: Jiajie
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import  MLPRegressor
from sklearn.linear_model import LinearRegression
import functools
from multiprocessing import Pool
import time
import joblib
import sys
from scipy import stats
sys.path.append("..")
import FunctionsZ
import os

def FeaLabel_name():
    # text-related features
    order = ['Sent_id', 'IA_ID', 'WordIdInSent']
    text = ['WordFreqBookWiki_srilm', 'Word_len', 'WordSurprisalBookWiki_GPT']
    
    QuesUnreFea_name_srilm = order + text
    QuesUnreFea_name_srilm = [
        QuesUnreFea_name_tmp + '_lp' 
        for QuesUnreFea_name_tmp in QuesUnreFea_name_srilm
        ]
    
    # question relevance
    QuesReFea_name_lp = ['RationaleTime5_lp']
    AllFea_name = QuesUnreFea_name_srilm + QuesReFea_name_lp
    
    # data loading
    Fea_name = { 
        'QuesUnreFeaP': QuesUnreFea_name_srilm,
        'QuesReFea_name': QuesReFea_name_lp,
        'AllFea': AllFea_name
        }
    #label
    SAR_Atten = ['SAR_Atten_lp']
    EyeLabel_name = [
        'IA_DWELL_TIME_lp', 
        'IA_FIRST_RUN_DWELL_TIME_lp', 
        'IA_RUN_COUNT_Del0_lp'
        ]
    AttenLabel_name = [
        att_seg_tmp + '-' + str(head_i) + '_lp'
        for att_seg_tmp in ['CLS2P'] 
        for head_i in range(144)
        ]
    Label_name = SAR_Atten + EyeLabel_name + AttenLabel_name
    return Fea_name, Label_name

def Modeling(paras, Fea_name, Label_name, fea_label, cv_num, test_ratio, Type, out_fea=None):
    t_B = time.time()
    pca_ratio = paras['pca_ratio']
    fea_ctg_name_tmp = paras['fea_ctg_name']
    rgs_tmp = paras['rgs_name']
    Label = fea_label[Label_name]
    Label_Type = Label.columns
    Labels_np = Label.values.astype(float)
    ######---------------Feature extraction-----------------#####
    Feas_np = fea_label[Fea_name[fea_ctg_name_tmp]].values.astype(float)
    
    if out_fea is not None:
        Feas_out_np = fea_label[Fea_name[out_fea]].values.astype(float)
    else:
        Feas_out_np = None
    
    # split by doc
    np.random.seed(0)
    Doc_index = fea_label['Doc_ID'] + 0
    Doc_ID = Doc_index.unique()
    Doc_ID = np.sort(Doc_ID)
    test_size = int(test_ratio*len(Doc_ID))
    # randomize
    doc_ID_rand = np.random.choice(Doc_ID, size=len(Doc_ID), replace=False)    
    # cross validation
    model_comparison = pd.DataFrame()
    cv_ind = paras['cv_ind']
    doc_ID_test = doc_ID_rand[(cv_ind*test_size):((cv_ind+1)*test_size)]
    test_flag = Doc_index.isin(doc_ID_test)
    doc_index_test = Doc_index[test_flag] + 0
    ######---------------Modeling-----------------#####
    for Label_ind in range(Labels_np.shape[-1]):
            
        Labels_np_tmp = Labels_np[:, Label_ind].reshape((-1, 1))
        Labels_train = Labels_np_tmp[test_flag!=True] + 0
        Labels_test = Labels_np_tmp[test_flag] + 0
        
        # regress out
        if out_fea is not None:
            Feas_train_out = Feas_out_np[test_flag!=True, :] + 0
            Feas_test_out = Feas_out_np[test_flag, :] + 0
            #Preprocessing
            (Feas_train_out, Feas_test_out, Labels_train, Labels_test, 
             scaler_fea_out1, Reducter_out, scaler_fea_out2) = FunctionsZ.Prepocessing_func(
                 Feas_train_out, Feas_test_out, Labels_train, Labels_test, pca_ratio)
        
        # residual
        Feas_train = Feas_np[test_flag!=True, :] + 0
        Feas_test = Feas_np[test_flag, :] + 0
        #Preprocessing
        (Feas_train, Feas_test, Labels_train, Labels_test,
         scaler_fea1, Reducter, scaler_fea2) = FunctionsZ.Prepocessing_func(
                 Feas_train, Feas_test, Labels_train, Labels_test, pca_ratio)
             
        # Normalizing
        scaler_label = None
        Labels_train = Labels_train.reshape((-1))
        Labels_test = Labels_test.reshape((-1))
        
        if rgs_tmp=='MLP':
            if out_fea is not None:
                Regressor = [
                    MLPRegressor(
                        alpha=0.01, hidden_layer_sizes=(10, 4), 
                        random_state=0, max_iter=1000, 
                        early_stopping=True, n_iter_no_change=50), 
                    MLPRegressor(
                        alpha=0.01, hidden_layer_sizes=(10, 4), 
                        random_state=0, max_iter=1000, 
                        early_stopping=True, n_iter_no_change=50), 
                    ]
            else:
                Regressor = [
                    MLPRegressor(
                        alpha=0.01, hidden_layer_sizes=(10, 4), 
                        random_state=0, max_iter=1000, 
                        early_stopping=True, n_iter_no_change=50)
                    ]
        elif rgs_tmp=='LR':
            if out_fea is not None:
                Regressor = [LinearRegression(), LinearRegression()]
            else:
                Regressor = [LinearRegression()]
            
        if out_fea is not None:
            model_comparison_tmp = FunctionsZ.regressor_residual_func(
                Feas_train_out, Feas_test_out, Feas_train, 
                Feas_test, Labels_train, Labels_test, Regressor)
        else:
            model_comparison_tmp = FunctionsZ.regressor_comparison_func(
                    Feas_train, Feas_test, Labels_train, Labels_test, Regressor)   
            
        print('*'*20 + Type + '-' + Label_Type[Label_ind] + '*'*25)
        print(paras)
        print('PCA: ' + str(Feas_train.shape[1]) + '/' + str(Feas_np.shape[1]))
        print('Label: ' + Label_Type[Label_ind])
        
        # eval by docs 
        if out_fea is not None:
            rgs_out = model_comparison_tmp['RegressorOut'].values[0]
            Labels_test_pred_out = rgs_out.predict(Feas_test_out)
            Labels_test = Labels_test - Labels_test_pred_out
            rgs_resi = model_comparison_tmp['RegressorResi'].values[0]
            Labels_test_pred = rgs_resi.predict(Feas_test)
            assert (Labels_test_pred[:100] == 
                    np.array(model_comparison_tmp['y_test_pred_resi'].values[0])).all()
        else:
            # eval by docs 
            rgs = model_comparison_tmp['Regressor'].values[0]
            Labels_test_pred = rgs.predict(Feas_test)
            
        test_coefs = []
        for doc_test_tmp in doc_ID_test:
            doc_test_flag_tmp = doc_index_test==doc_test_tmp
            label_test_pred_tmp = Labels_test_pred[doc_test_flag_tmp] + 0
            Labels_test_tmp = Labels_test[doc_test_flag_tmp] + 0
            test_coef_tmp, _ = stats.pearsonr(
                Labels_test_tmp, label_test_pred_tmp)
            test_coefs.append(test_coef_tmp)
            
        model_comparison_tmp['rgs_name'] = rgs_tmp
        model_comparison_tmp['target_name'] = Label_Type[Label_ind]
        model_comparison_tmp['Fea'] = fea_ctg_name_tmp
        model_comparison_tmp['FeaOut'] = out_fea if out_fea is not None else None
        
        model_comparison_tmp['fea_out_name'] = [Fea_name[out_fea]] if out_fea is not None else None
        model_comparison_tmp['fea_name'] = [Fea_name[fea_ctg_name_tmp]]
        
        model_comparison_tmp['pca_ratio'] = pca_ratio
        model_comparison_tmp['PCA'] = Feas_train.shape[1]
        model_comparison_tmp['Raw'] = Feas_np.shape[1]
        
        model_comparison_tmp['Scaler_fea1'] = scaler_fea1
        model_comparison_tmp['Reducter'] = Reducter
        model_comparison_tmp['Scaler_fea2'] = scaler_fea2
        
        model_comparison_tmp['Scaler_fea_out1'] = scaler_fea_out1 if out_fea is not None else None
        model_comparison_tmp['Reducter_out'] = Reducter_out if out_fea is not None else None
        model_comparison_tmp['Scaler_fea_out2'] = scaler_fea_out2 if out_fea is not None else None
        
        model_comparison_tmp['Scaler_label'] = scaler_label
        model_comparison_tmp['TestCoef_docs'] = [test_coefs]
        model_comparison_tmp['CV_ind'] = cv_ind
        model_comparison_tmp['doc_ID_test'] = [doc_ID_test]
        model_comparison = model_comparison.append(model_comparison_tmp)
        
        if out_fea is not None:
            print('TestCoef: ' + str(model_comparison_tmp.TestCoefResi.values))
        else:
            print('TestCoef: ' + str(model_comparison_tmp.TestCoef.values))
        print('TestCoef_doc: ' + str(np.mean(test_coefs)))
        print('Time: ' + str(time.time()-t_B))
    return model_comparison
    
#%%Classification
if __name__ == '__main__':
    FunctionsZ.FontSetting()
    
    t_B_all = time.time()
    pool = Pool(36)
    fake_flag = False
    att_path = '../NLP/AttentionMap'
    regresses = ['full', 'VT_Resi']
    
    for regresses_tmp in regresses:
        for reason_method in ['model_fine', 'model_pre', 'model_random']:
            for bert_tmp in ['bert', 'albert', 'roberta']:
                for Type in ['Mainly', 'Title', 'GPurpose', 'LPurpose', 'Fact', 'Limply']:
                    bert_model = '/' + reason_method + '/' + bert_tmp + '/'
                    type_b_time = time.time()
                    # Setting
                    pca_ratio = [1]
                    cv_num = 5
                    test_ratio = 1/cv_num
                    
                    # data loading 
                    save_fealabel_folder = f'./AttentionMap{bert_model}FeaLabel' 
                    FeaLabel_types_file = save_fealabel_folder + '/800_FeaLabel_bert_230624_subjs.pickle'
                    FeaLabel_bert_info = pickle.load(open(FeaLabel_types_file, 'rb'))
                    FeaLabel2_ = FeaLabel_bert_info['FeaLabel2']
                    FeaLabel2_['Type'] = FeaLabel2_.Doc_ID.map(lambda x: x.split('_')[0])
                    FeaLabel2_['Doc_ID'] = FeaLabel2_.Doc_ID.map(lambda x: x.split('_')[-1]).astype(int)
                    FeaLabel2 = FeaLabel2_.loc[FeaLabel2_.Type==Type, :]
                    Fea_name, Label_name = FeaLabel_name()
                    
                    # select subjects
                    subj_id_save = ''
                    for label_tmp in Label_name:
                        label_tmp_subj = (
                            label_tmp + '_subj' + str(subj_id_save) 
                            if subj_id_save!='' else label_tmp)
                        FeaLabel2[label_tmp] = FeaLabel2[label_tmp_subj]
                        
                    # select useful columns
                    FeaLabel_bert = pd.DataFrame()
                    for columns_tmp in (Fea_name['AllFea'] + Label_name + ['Doc_ID']):   
                        FeaLabel_bert[columns_tmp] = FeaLabel2[columns_tmp]
                    
                    # delete nan
                    print(f'=============={len(FeaLabel_bert.index)}==============')
                    FeaLabel_bert = FeaLabel_bert.replace([np.inf, -np.inf], np.nan)
                    FeaLabel_bert = FeaLabel_bert.dropna(axis=0, how='any')
                    print(f'--------------{len(FeaLabel_bert.index)}--------------')
                    
                    # regress out text influences or not
                    if regresses_tmp == 'full': 
                        out_fea = None
                    elif regresses_tmp == 'VT_Resi': 
                        out_fea = 'QuesUnreFeaP'
                    # model setting
                    rgs_name = ['LR']
                    paras = [{'pca_ratio': pca_ratio_tmp, 'fea_ctg_name': fea_ctg_tmp, 
                              'rgs_name':rgs_tmp, 'cv_ind':cv_ind} 
                    for pca_ratio_tmp in pca_ratio
                    for fea_ctg_tmp in Fea_name.keys()
                    for rgs_tmp in rgs_name
                    for cv_ind in range(cv_num)]
                    
                    # modeling
                    print(reason_method + bert_tmp + Type + 'modeling')
                    model_comparison = pool.map(
                        functools.partial(
                            Modeling, Fea_name=Fea_name, Label_name=Label_name,
                            fea_label=FeaLabel_bert, cv_num=cv_num, 
                            test_ratio=test_ratio, Type=Type, 
                            out_fea=out_fea
                            ), paras)
                    
                    rgs_result = pd.DataFrame()
                    for model_comparison_tmp in model_comparison:
                        rgs_result = rgs_result.append(model_comparison_tmp)
                        
                    save_predacc_folder = f'./AttentionMap{bert_model}PredAcc' 
                    if not os.path.exists(save_predacc_folder):
                        os.makedirs(save_predacc_folder)
                    joblib.dump(rgs_result, open(
                            save_predacc_folder + f'pred_atten_{regresses_tmp}.pickle', 'wb'))
                    print('*'*50 + Type + str(time.time() - type_b_time) + '*'*50)
    pool.close()
    pool.join()
    print(time.time() - t_B_all)