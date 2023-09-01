# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:28:02 2023

export data for LMM

@author: bme106
"""


import pickle
import numpy as np
from tqdm import tqdm

def FeaName():
    # QuesUnreFea
    visual_layout = ['Para_id', 'RowID', 'IA_LEFT', 'RowIdInPara']
    
    text_smp = [ 
        'WordFreqBookWiki_srilm', 'Word_len', 'WordSurprisalBookWiki_GPT'
        ]
    
    # QuesReFea_name
    QuesReFea_name_lp = ['RationaleTime5']
    
    # SAR
    SAR_Atten = ['SAR_Atten_lp']
    
    # DNN
    AttenFea = [
        'CLS2P-' + str(head_i) + '_lp'
        for head_i in range(144)
        ]
    
    Fea_name = (
        visual_layout + text_smp + QuesReFea_name_lp + 
        SAR_Atten + AttenFea
        )
    return Fea_name


def columns_rename():
    dict_rename = {}
    for fea_tmp in Fea_name:
        if fea_tmp == 'WordFreqBookWiki_srilm':
            dict_rename_tmp = {fea_tmp: 'WordFreq'}
        elif fea_tmp == 'WordSurprisalBookWiki_GPT':
            dict_rename_tmp = {fea_tmp: 'Surprisal'}
        elif fea_tmp == 'RationaleTime5':
            dict_rename_tmp = {fea_tmp: 'Rationale'}
        else:
            dict_rename_tmp = {fea_tmp: fea_tmp.replace('CLS2P-', 'DNN').strip('_lp')}
        dict_rename.update(dict_rename_tmp)
    return dict_rename


# %%
if __name__ == '__main__':
    for reason_method in ['model_fine', 'model_pre', 'model_random']:
        for bert_tmp in ['bert']:
            bert_model = '/' + reason_method + '/' + bert_tmp + '/'
            save_fealabel_folder = f'../AttentionMap{bert_model}FeaLabel' 
            FeaLabel_types_file = save_fealabel_folder + '/800_FeaLabel_bert_230624_subjs.pickle'
            
            FeaLabel_bert_info = pickle.load(open(FeaLabel_types_file, 'rb'))
            FeaLabel2 = FeaLabel_bert_info['FeaLabel2']
            
            Fea_name = FeaName()
            Fea_name = ['Doc_ID', 'IA_ID', 'Word'] + Fea_name
            
            subj_ids = np.arange(25)
            col_rename = columns_rename()
            for subj_tmp in tqdm(subj_ids):
                print(subj_tmp)
                
                fea_label_r_tmp = FeaLabel2[Fea_name+[f'IA_DWELL_TIME_subj{subj_tmp}']]
                fea_label_r_tmp['Subj_ID'] = subj_tmp 
                fea_label_r_tmp['Type'] = fea_label_r_tmp.Doc_ID.apply(lambda x:x.split('_')[0])
                # rename columns
                read_rename = {f'IA_DWELL_TIME_subj{subj_tmp}': 'read_duration'}
                fea_label_r_tmp = fea_label_r_tmp.rename(columns=read_rename)
                fea_label_r_tmp = fea_label_r_tmp.rename(columns=col_rename)
                fea_label_r_tmp['WordFreq'] = -fea_label_r_tmp['WordFreq']
                fea_label_r_tmp['IA_LEFT'] = fea_label_r_tmp['IA_LEFT'] / 1920
                
                fea_label_r_tmp.to_csv(
                    f'./Data/subjs/{reason_method}_{bert_tmp}_DataSubj_{subj_tmp}.csv'
                )
            
                