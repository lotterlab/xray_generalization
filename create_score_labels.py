import os
import pandas as pd

from pycrumbs import tracked # add pycrumbs to track inference runs (the records are saved with the dataset csv files)
from utils import read_prediction_df, read_dataset_df, save_score_label_df

@tracked(directory_parameter='record_dir')
def create_score_labels(dataset,
                        path_sign,
                        train_split, 
                        splits: list,
                        prediction_target: str,
                        model_names_dict, # contains the names of the all model versions 
                        path_templates_dict,
                        record_dir,
                        median_split_dataset = None, # split used for training of the score model 
                        mode = 'raw_score', # ['raw_score', 'rank'] compute the score difference based on raw or ranked mean prediction
                        project_root = '/lotterlab/users/khoebel/xray_generalization'
                        ):
    # sort the splits list s.t. the median_split_dataset (if defined) is first
    # print(splits)
    if median_split_dataset is not None: 
        assert (median_split_dataset in splits), 'median split dataset needs to be in splits list'
        splits.insert(0, splits.pop(splits.index(median_split_dataset)))
    # print(splits)

    for split in splits:
        dataset_pred_dfs = list()
        for model_train_dataset in ['mmc','cxp']:
            temp_df_list = list()
            for model in model_names_dict[model_train_dataset]:
                temp_df_list.append(read_prediction_df(dataset, train_split, model, split, prediction_target, path_template=path_templates_dict['pred'], project_root=project_root))
                
            # compare the order of rows in all dataframes (to make sure that I can concenate in the next step)
            order_comparison = [temp_df_list[0]['Path'].equals(temp_df['Path']) for temp_df in temp_df_list[1:]]
            assert all(order_comparison)
            
            # concatenate the prediction dataframes for all versions of the model
            temp_df = pd.concat(temp_df_list,axis=1, ignore_index=False)
            assert temp_df['Pred_'+path_sign.capitalize()].shape[1]==3, 'prediction columns do not have the same name'
            
            # average predictions
            temp_df['mean_Pred_'+model_train_dataset] = temp_df['Pred_'+path_sign.capitalize()].mean(axis=1)
            
            # remove duplicate columns
            temp_df = temp_df.loc[:,~temp_df.columns.duplicated()].copy()
            
            # only keep the columns we need
            temp_df = temp_df[['Path','mean_Pred_'+model_train_dataset, path_sign.capitalize()]]

            dataset_pred_dfs.append(temp_df)
            
        # combine the predictions for MMC and CXP pathology models 
        order_comparison = [dataset_pred_dfs[0]['Path'].equals(temp_df['Path']) for temp_df in dataset_pred_dfs[1:]]
        assert all(order_comparison)
        dataset_pred_df = pd.concat(dataset_pred_dfs, axis=1)
        dataset_pred_df = dataset_pred_df.loc[:,~dataset_pred_df.columns.duplicated()].copy() # remove duplicate columns
        if mode == 'raw_score': 
            dataset_pred_df['Score_Diff'] = dataset_pred_df['mean_Pred_cxp'] - dataset_pred_df['mean_Pred_mmc'] # calculate score difference (cxp - mmc)
        elif mode == 'rank':
            dataset_pred_df['mmc_rank'] = dataset_pred_df['mean_Pred_mmc'].rank(pct=True, ascending=True)
            dataset_pred_df['cxp_rank'] = dataset_pred_df['mean_Pred_cxp'].rank(pct=True, ascending=True)
            dataset_pred_df['Score_Diff'] = dataset_pred_df['cxp_rank'] - dataset_pred_df['mmc_rank'] # calculate difference based on ascending ranks (cxp - mmc)
        else:
            raise ValueError(f"Invalid value for mode: {mode}. Mode must be 'rank' or 'raw'.")

        
        # binarize 
        path_sign_idx = dataset_pred_df[path_sign.capitalize()]==1
        if (median_split_dataset is None) or (split == median_split_dataset):
            median_score = dataset_pred_df.loc[path_sign_idx,'Score_Diff'].median()
        print(split, median_score)
        dataset_pred_df['Higher_Score'] = ['CXP' if a >= median_score else 'MMC' for a in dataset_pred_df['Score_Diff'].values]
        
        # read in the pathology dataset csv
        if dataset == 'cxp':
            file_name_modifier = ''
            dataset_pred_df['Path'] = [os.path.join(*a.split('/')[3:]) for a in dataset_pred_df['Path']]
            join_col = 'Path'
        elif dataset == 'mmc':
            file_name_modifier = 'meta_'
            dataset_pred_df['dicom_id'] = [a.split('/')[-1][:-4] for a in dataset_pred_df['Path']]
            join_col = 'dicom_id'
        
        orig_data_df = read_dataset_df(dataset, train_split, file_name_modifier, split,prediction_target, path_template=path_templates_dict['dataset'], project_root=project_root)

        # drop Pneumothorax column (want to keep the one for the prediction dataframe)
        try: 
            orig_data_df = orig_data_df.drop(path_sign.capitalize(), axis=1)
        except:
            pass
        
        # duplicate original index column
        orig_data_df['Image_ID'] = orig_data_df.index
        
        # merge score labels and original data
        dataset_pred_df = dataset_pred_df.merge(orig_data_df,how='inner', right_on=join_col, left_on=join_col)
        
        # only keep rows with valid path_sign labels 
        dataset_pred_df = dataset_pred_df.loc[dataset_pred_df[path_sign.capitalize()]==1]
                                              
        # save 
        save_score_label_df(dataset_pred_df,dataset,train_split,split,path_sign, file_name_modifier,  path_template=path_templates_dict['save'], project_root=project_root)


if __name__ == "__main__":
        

        project_root = '/lotterlab/users/khoebel/xray_generalization'


        path_templates_dict = {'pred': 'data/splits/{0}/{1}/{4}/prediction_dfs/{2}/pred_{0}-{3}_df.csv',
                             'dataset': 'data/splits/{0}/{1}/{4}/{2}{3}.csv',
                             'save': 'data/splits/{0}/{1}/{2}/{3}{4}.csv'
                             }
      
        train_split = str(0.7)
        path_sign = 'pneumothorax'


        model_names_dict = {'cxp': ['cxp_densenet_pretrained_v2-best',
                                   'cxp_densenet_pretrained_v3-best',
                                   'cxp_densenet_pretrained_v4-best'],
                           'mmc': ['mimic_densenet_pretrained_v2-best',
                                   'mimic_densenet_pretrained_v3-best',
                                   'mimic_densenet_pretrained_v4-best']}


        splits = ['test', 'val']
        prediction_target = 'pathology'
        mode = 'rank'

        
        '''project_root = '/lotterlab/users/khoebel/xray_generalization'

        path_templates_dict = {'pred': 'data/splits/{0}/{1}/{4}/prediction_dfs/{2}/pred_{0}-{3}_df.csv',
                              'dataset': 'data/splits/{0}/{1}/{4}/{2}{3}.csv',
                              'save': 'data/splits/{0}/{1}/{2}/{3}{4}.csv'
                              }
        
        train_split = str(0.35)
        path_sign = 'pneumothorax'
        prediction_target = 'pathology'

        model_names_dict = {'cxp': ['cxp_densenet_pretrained_0.35-best', 
                                    'cxp_densenet_pretrained_0.35_seed_1-best', 
                                    'cxp_densenet_pretrained_0.35_seed_1-best'], 
                            'mmc': ['mmc_densenet_pretrained_0.35-best', 
                                    'mmc_densenet_pretrained_0.35_seed_1-best', 
                                    'mmc_densenet_pretrained_0.35_seed_2-best']}

        splits = ['test', 'val', 'train_score']'''

        for dataset in ['cxp', 'mmc']:
        
            record_dir = os.path.join(project_root, 'data','splits', dataset, train_split, path_sign)
            print(record_dir)
            create_score_labels(dataset,
                        path_sign,
                        train_split, 
                        splits,
                        prediction_target ,
                        model_names_dict, # contains the names of the all model versions 
                        path_templates_dict,
                        record_dir,
                        median_split_dataset = 'val',
                        mode = 'rank',
                        project_root = '/lotterlab/users/khoebel/xray_generalization'
                        )