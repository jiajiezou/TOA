# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:41:10 2020

Predict eye measures of human reading with hand-crafted features 
(i.e., layout features, word features, and question relevance) and DNN attention

@author: Jiajie Zou
"""
import pickle
import numpy as np
import pandas as pd
import functools
from multiprocessing import Pool
import time
import joblib
import sys
from scipy import stats
sys.path.append("..")
import FunctionsZ
import os
from sklearn.linear_model import LinearRegression

def FeaLabel_name():
    # -----hand-crafted features
    # layout features
    visual_layout = ['Para_id_lp', 'RowID_lp', 'IA_LEFT_lp', 'RowIdInPara_lp']
    # word features
    text = ['WordFreqBookWiki_srilm_lp', 'Word_len_lp', 
            'WordSurprisalBookWiki_GPT_lp']
    # question relevance
    QuesReFea_name_lp = ['RationaleTime5_lp']
    
    # -----computational models
    # orthographic model
    editting_mean = ['EdittingScoreMean_lp']
    # semantic model
    glove_mean = ['GloveScoreMean_lp']
    # SAR
    SAR_Atten = ['SAR_Atten_lp']
    # transformer-based model 
    AttenFea = [
        att_seg_tmp + '-' + str(head_i) + '_lp'
        for att_seg_tmp in ['CLS2P'] 
        for head_i in range(144)
        ]
    
    # -----integrated features 
    # visual + text
    bot_up_fea = visual_layout + text
    # visual + text + relevance
    manual_fea = visual_layout + text + QuesReFea_name_lp
    # all features
    AttenAllFea_lp = (
        visual_layout + text + QuesReFea_name_lp + 
        SAR_Atten + AttenFea + glove_mean + editting_mean
        )
    
    # feature dict
    Fea_name = {
        'visual_layout': visual_layout, 
        'text_smp': text, 
        'QuesUnreFea': bot_up_fea, 
        'QuesReFea_name': QuesReFea_name_lp, 
        'Manual_name_lp':manual_fea, 
        
        'SAR_Atten':SAR_Atten, 
        'AttenFea':AttenFea, 
        'GloveMean':glove_mean,
        'EditMean':editting_mean,
        'AllFea':AttenAllFea_lp
        }
    # dict of eye measures
    Label_name = [
        'IA_DWELL_TIME_lp', 
        'IA_FIRST_RUN_DWELL_TIME_lp', 
        'IA_RUN_COUNT_Del0_lp'
        ]
    
    return Fea_name, Label_name

def Modeling(paras, Fea_name, fea_label, cv_num, test_ratio, out_fea=None):
    t_B = time.time()
    pca_ratio = paras['pca_ratio']
    label_ctg_name_tmp = paras['label_ctg_name']
    fea_ctg_name_tmp = paras['fea_ctg_name']
    rgs_tmp = paras['rgs_name']
    Label = fea_label[label_ctg_name_tmp]
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
    Doc_index = fea_label['Doc_ID']
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
    doc_index_test = Doc_index[test_flag]
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
        print('*'*20 + '-' + Label_Type[Label_ind] + '*'*25)
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
            
            # write the predicted attention to ia report
            pred_columns = (
                'Pred~' + Label_Type[Label_ind] + '~' 
                + fea_ctg_name_tmp + '~' + str(out_fea)
                )
            fea_label.loc[fea_label.Doc_ID==doc_test_tmp, pred_columns
                          ] = label_test_pred_tmp
        
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
    pool = Pool(36)
    fake_flag = False
    att_path = '../NLP/AttentionMap'
    regresses = ['full', 'V_Resi', 'VT_Resi', 'VTR_Resi']
    exs = ['800', 'Mixed', 'MixedNative', 'MixedGoalL2']
    Types = ['Mainly', 'Title', 'GPurpose', 'LPurpose', 'Fact', 'Limply']
    
    for meas_test in ['RTonly', 'All']:
        for ex_tmp in exs:
            for regresses_tmp in regresses:
                state = '_' + ex_tmp + '_' + regresses_tmp
                for reason_method in ['model_fine', 'model_pre', 'model_random']:
                    for bert_tmp in ['bert', 'albert', 'roberta']:
                        bert_model = '/' + reason_method + '/' + bert_tmp + '/'
                        type_b_time = time.time()
                        
                        # setting
                        pca_ratio = [1]
                        cv_num = 5
                        test_ratio = 1/cv_num
                        
                        # data loading 
                        save_fealabel_folder = f'./AttentionMap{bert_model}FeaLabel' 
                        FeaLabel_types_file = save_fealabel_folder + f'/{ex_tmp}_FeaLabel_bert_230624_subjs.pickle'
                        FeaLabel_bert_info = pickle.load(open(FeaLabel_types_file, 'rb'))
                        FeaLabel2 = FeaLabel_bert_info['FeaLabel2']
                        
                        Fea_name, Label_name = FeaLabel_name()
                        if meas_test == 'RTonly':
                            Label_name = ['IA_DWELL_TIME_lp']
                        
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
                        
                        # regress out visual contributions or not
                        if 'V_Resi' in state: 
                            out_fea = 'visual_layout'
                        elif 'VT_Resi' in state: 
                            out_fea = 'QuesUnreFea'
                        elif 'VTR_Resi' in state: 
                            out_fea = 'Manual_name_lp'
                        else:
                            out_fea = None
                            
                        #model setting
                        rgs_name = ['LR']
                        paras = [
                            {'pca_ratio': pca_ratio_tmp, 
                              'label_ctg_name': [label_ctg_tmp], 
                              'fea_ctg_name': fea_ctg_tmp, 
                              'rgs_name':rgs_tmp, 
                              'cv_ind':cv_ind
                              } 
                        for pca_ratio_tmp in pca_ratio
                        for label_ctg_tmp in Label_name
                        for fea_ctg_tmp in Fea_name.keys()
                        for rgs_tmp in rgs_name
                        for cv_ind in range(cv_num)
                        ]
                        
                        #modeling reslut
                        print(reason_method + bert_tmp + 'modeling')
                        
                        model_comparison = pool.map(
                            functools.partial(
                                Modeling, 
                                Fea_name=Fea_name,  
                                fea_label=FeaLabel_bert, 
                                cv_num=cv_num, 
                                test_ratio=test_ratio, 
                                out_fea=out_fea), paras)
                        
                        rgs_result = pd.DataFrame()
                        for model_comparison_tmp in model_comparison:
                            rgs_result = rgs_result.append(model_comparison_tmp)
                            
                        save_predacc_folder = f'./AttentionMap{bert_model}PredAcc' 
                        if not os.path.exists(save_predacc_folder):
                            os.makedirs(save_predacc_folder)
                        joblib.dump(
                            {'rgs_result':rgs_result, 'FeaLabel_bert':FeaLabel_bert}, 
                            open(save_predacc_folder + f'/pred_eye_230624{state}_{meas_test}_subjs.pickle', 'wb')
                            )
                        print('*'*50 + str(time.time() - type_b_time) + '*'*50)
                    
    pool.close()
    pool.join()
    # %%
    regresses = ['full', 'V_Resi', 'VT_Resi', 'VTR_Resi']
    exs = ['800', 'Mixed', 'MixedNative', 'MixedGoalL2']
    states = [
        '_' + ex_tmp + '_' + regresses_tmp 
        for ex_tmp in exs for regresses_tmp in regresses
        ]
    pred_perm = pd.DataFrame(
        columns=['State', 'ReasonMethod', 'Model', 
                  'Type', 'Target', 'Feature', 'Doc_ID', 'Performance']
        )
    for state in states:
        types = ['LPurpose', 'Fact', 'Limply', 'Mainly', 'Title', 'GPurpose']
        for reason_method in ['model_fine', 'model_pre', 'model_random']:
            for bert_tmp in ['bert', 'albert', 'roberta']:
                bert_model = '/' + reason_method + '/' + bert_tmp + '/'
                save_predacc_folder = f'./AttentionMap{bert_model}PredAcc'
                print(state + bert_model)
                # -----------------------comparsion between types--------------------#
                bert_pred_info_rt = joblib.load(open(save_predacc_folder + f'/pred_eye_230624{state}_RTonly_subjs.pickle', 'rb'))
                rgs_result_types_rt = bert_pred_info_rt['rgs_result']
                
                bert_pred_info_all = joblib.load(open(save_predacc_folder + f'/pred_eye_230624{state}_All_subjs.pickle', 'rb'))
                rgs_result_types_all = bert_pred_info_all['rgs_result']
                
                targets = [
                    'IA_DWELL_TIME_lp', 
                    'IA_FIRST_RUN_DWELL_TIME_lp', 
                    'IA_RUN_COUNT_Del0_lp'
                    ]
                fea_names = [
                    'visual_layout', 'text_smp', 'QuesReFea_name', 
                    'EditMean', 'GloveMean', 'SAR_Atten', 'AttenFea'
                    ]
                for target in targets:
                    for fea_name in fea_names:
                        if target == 'IA_DWELL_TIME_lp':
                            rgs_result_tmp = rgs_result_types_rt.loc[
                                    (rgs_result_types_rt.target_name==target) & 
                                    (rgs_result_types_rt.Fea==fea_name) & 
                                    (rgs_result_types_rt.rgs_name=='LR'), :]
                        else:
                            rgs_result_tmp = rgs_result_types_all.loc[
                                    (rgs_result_types_all.target_name==target) & 
                                    (rgs_result_types_all.Fea==fea_name) & 
                                    (rgs_result_types_all.rgs_name=='LR'), :]
                            
                        assert len(rgs_result_tmp.index) == 5
                        
                        # per doc
                        rgs_doc_id_list = rgs_result_tmp['doc_ID_test'].tolist()
                        rgs_doc_id_list_flat = np.concatenate(rgs_doc_id_list)
                        
                        rgs_result_list = rgs_result_tmp['TestCoef_docs'].tolist()
                        rgs_result_list_flat = [
                            rgs_result_tmp 
                            for rgs_result_list_tmp in rgs_result_list 
                            for rgs_result_tmp in rgs_result_list_tmp
                            ]
                        
                        for Type in types:
                            doc_id_type = [
                                doc_id_tmp for doc_id_i, doc_id_tmp in enumerate(rgs_doc_id_list_flat) 
                                if Type in doc_id_tmp
                                ]
                            rgs_result_list_flat_type = [
                                rgs_result_list_flat[doc_id_i] for doc_id_i, doc_id_tmp in enumerate(rgs_doc_id_list_flat) 
                                if Type in doc_id_tmp
                                ]
                            
                            perm_doc_id_tmp = ', '.join(doc_id_type)
                            perm_tmp = ', '.join(np.round(np.array(rgs_result_list_flat_type), 5).astype(str))
                            
                            pred_perm.loc[len(pred_perm.index), :] = [
                                state, reason_method, bert_tmp, Type, target, fea_name, perm_doc_id_tmp, perm_tmp
                                ]
                            print(f'{target}, {fea_name}, {Type}, {len(doc_id_type)}')
    pred_perm.to_csv('./PredEyeCSV/EyePred.csv')
    
    
    
    
    
    
    
    
    
                