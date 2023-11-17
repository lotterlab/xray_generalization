"""
combine the inference outputs with the ground truth in one df

original code in results.py (/lotterlab/users/jfernando/project_1/scripts)

date separated from inference: 11/17/2023
"""

import os
import pandas as pd
import sys

from pycrumbs import tracked # add pycrumbs to track inference runs (the records are saved with the dataset csv files)

sys.path.append('/lotterlab/users/khoebel/xray_generalization/scripts/torchxrayvision/')
import torchxrayvision as xrv

sys.path.append('/lotterlab/users/jfernando/project_1/scripts/')
from data_utils import create_mimic_isportable_column
from constants import PROJECT_DIR # KVH: MODEL_SAVE_DIR, overwriting some of the constants at other points in this script




def load_dataset(dataset_name, split):
    if dataset_name == 'cxp':
        # this_path = f'/lotterlab/lotterb/project_data/bias_interpretability/cxp_cv_splits/version_0/{split}.csv' # KVH: commented out because split paths will change for each experiment
        this_path = os.path.join(PROJECT_DIR, split +'.csv')
        dataset = xrv.datasets.CheX_Dataset(
            imgpath='',
            csvpath=this_path,
            transform=[], data_aug=None, unique_patients=False, views='all' ,use_no_finding=True)
    else:
        # csvpath = f'/lotterlab/lotterb/project_data/bias_interpretability/mimic_cv_splits/version_0/cxp-labels_{split}.csv' # KVH: commented out because split paths will change for each experiment
        csvpath = os.path.join(PROJECT_DIR,'cxp-labels_'+ split +'.csv')
        metacsvpath = csvpath.replace('cxp-labels', 'meta')
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath='',
            csvpath=csvpath,
            metacsvpath=metacsvpath,
            transform=[], 
            data_aug=None, 
            unique_patients=False, 
            views='all', 
            use_no_finding=True)

    return dataset


def load_pred_df(model_name, dataset_name, split, checkpoint_name='best', merge_labels=True, window_width=None, resize_factor=None, prediction_mode='pathology'):
    tag = ''
    if window_width or resize_factor:
        if window_width:
            tag += f'-window{window_width}'
        if resize_factor:
            tag += f'-initresize{resize_factor}_midcrop'
    pred_path = os.path.join(PROJECT_DIR, 'prediction_dfs', model_name + '-' + checkpoint_name, dataset + '-' + split +'.csv')
    # os.path.join(PROJECT_DIR + 'prediction_dfs', model_name + '-best', dataset_name + '-' + split + tag + '.csv')      #load pred df
    pred_df = pd.read_csv(pred_path)

    if dataset_name == 'cxp':
        pred_df['orig_path'] = [p.replace('/lotterlab/datasets/', '') for p in pred_df.Path.values]
        study_ids = []
        for p in pred_df.Path.values:
            vals = p.split('/')
            study_ids.append(vals[-3] + '-' + vals[-2])
        pred_df['study_id'] = study_ids
    else:
        pred_df['dicom_id'] = [p.split('/')[-1][:-4] for p in pred_df.Path.values]
        pred_df['study_id'] = [p.split('/')[-2] for p in pred_df.Path.values]

    print(pred_df.head())

    if merge_labels:
        if prediction_mode=='pathology':
            # keep original structure
            xrv_dataset = load_dataset(dataset_name, split)
            if dataset_name == 'cxp':
                gt_df = pd.DataFrame(xrv_dataset.labels, columns=xrv_dataset.pathologies, index=xrv_dataset.csv.Path)
                merge_df = pd.merge(pred_df, gt_df, how='left', left_on='orig_path', right_index=True)
            else:
                gt_df = pd.DataFrame(xrv_dataset.labels, columns=xrv_dataset.pathologies, index=xrv_dataset.csv.dicom_id)
                merge_df = pd.merge(pred_df, gt_df, how='left', left_on='dicom_id', right_index=True)
                proc_map = xrv_dataset.csv[['PerformedProcedureStepDescription', 'dicom_id']].set_index('dicom_id')
                merge_df['PerformedProcedureStepDescription'] = merge_df.dicom_id.map(
                    proc_map['PerformedProcedureStepDescription'])
                merge_df = create_mimic_isportable_column(merge_df)

        elif prediction_mode=='higher_score':
            if dataset_name == 'cxp':
                gt_df = pd.read_csv(os.path.join(PROJECT_DIR,'{}.csv'.format(split)))
                print(gt_df.head())
                merge_df = pd.merge(pred_df, gt_df, how='left', left_on='orig_path', right_on='Path')
            elif dataset_name == 'mmc':
                gt_df = pd.read_csv(os.path.join(PROJECT_DIR,'meta_{}.csv'.format(split) ))
                merge_df = pd.merge(pred_df, gt_df, how='left', left_on='dicom_id', right_on='dicom_id')

        
        pred_path = os.path.join(PROJECT_DIR, 'prediction_dfs', model_name + '-' + checkpoint_name,'pred_'+dataset_name+'-'+split+'_df.csv')
        merge_df.to_csv(pred_path)
        return print('merge_df saved to: ', pred_path)
    else:
        pred_df.to_csv(pred_path+'/pred_'+dataset_name+'-'+split+'_df.csv')
        return print('print saved to: ', pred_path+'/pred_'+dataset_name+'-'+split+'_df.csv')



@tracked(directory_parameter='PROJECT_DIR')
def merge_predictions_with_gt(PROJECT_DIR:str, 
                      # model_dir_dict:dict, 
                      model_keys:list,
                      model_name_dict:dict,
                      splits:list,
                      prediction_mode:str,
                      checkpoint_name = 'best'
                      ):
    

    for model_key in model_keys:
    
        for model_name in model_name_dict[model_key]:
            print(model_name)
        
            for split in splits:
                
                load_pred_df(model_name,
                            dataset, 
                            split, 
                            checkpoint_name=checkpoint_name,
                            merge_labels=True, window_width=None, resize_factor=None,
                            prediction_mode=prediction_mode)
    



if __name__ == '__main__':


    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    prediction_mode = 'higher_score' # 'higher_score', 'pathology'
    
    splits = ['train_score', 'val', 'test']

    
     # .35 score model
    
    model_keys= ['mmc', 'cxp']
    

    model_name_dict = {'mmc': ['mmc_score_0.35_seed_1'], # list of all names of models for inference
                       'cxp': ['cxp_score_0.35_seed_1']
                       }
    

    project_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/data/splits/mmc/0.35/pneumothorax",
                        'cxp': "/lotterlab/users/khoebel/xray_generalization/data/splits/cxp/0.35/pneumothorax"
                        }
    

    # .7 score model
    
    '''model_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/models/mmc/0.7/pneumothorax",
                      # 'cxp': "/lotterlab/users/khoebel/xray_generalization/models/cxp/0.7/pneumothorax"
                      }
    

    model_name_dict = {'mmc': ['mmc_score_0.7_seed_1'], # list of all names of models for inference
                       # 'cxp': ['cxp_score_0.7_seed_1']
                       }
    

    project_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/data/splits/mmc/0.7/pneumothorax",
                        'cxp': "/lotterlab/users/khoebel/xray_generalization/data/splits/cxp/0.7/pneumothorax"
                        }'''

    # loop through project directories (i.e., datasets to run prediction on)
    for dataset in ['cxp', 'mmc']:
        PROJECT_DIR = project_dir_dict[dataset]
        print(PROJECT_DIR)
        merge_predictions_with_gt(PROJECT_DIR=PROJECT_DIR,
                          model_keys = model_keys, 
                          model_name_dict=model_name_dict,
                          splits=splits,
                          prediction_mode=prediction_mode,
                          checkpoint_name='best'
                          )
    