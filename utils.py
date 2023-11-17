import os 
import pandas as pd


def read_prediction_df(dataset:str,
                       train_split:str, 
                       model:str, 
                       split: str, 
                       prediction_target: str,
                       path_template = None,
                       project_root= '/lotterlab/users/khoebel/xray_generalization'
                      ):
    # reads in the prediction for the specified dataset and model 
    # dataset: prediction dataset ('mmc', 'cxp')
    # train_split: percentage of data the model has been trained on 
    # model: name of the model used to generate the predictions 
    # split: dataset split ('train', 'val','test')
    # prediction target: pathology or target label for higher score prediction (e.g. pneumothorax)

    if path_template is None:
        path_template = 'data/splits/{0}/{1}/{4}/prediction_dfs/{2}/pred_{0}-{3}_df.csv'
    path = os.path.join(project_root,path_template.format(dataset, train_split, model, split, prediction_target))
    
    return pd.read_csv(path)


def read_dataset_df(dataset:str,
                    train_split:str,
                    file_name_modifier: str,
                    split: str,
                    prediction_target: str,
                    path_template = 'data/splits/{0}/{1}/pathology/{2}{3}.csv',
                    project_root= '/lotterlab/users/khoebel/xray_generalization'
                   ):
    # reads in the dataset spreadsheet for the specified dataset and split 
    # dataset: prediction dataset ('mmc', 'cxp')
    # train_split: percentage of data the model has been trained on 
    # split: dataset split ('train', 'val','test')
    # prediction target: pathology or target label for higher score prediction (e.g. pneumothorax)
    # file_name_modifier: to comply with naming conventions between cxp and mmc ('', 'meta_')
     
    if path_template is None:
        path_template = 'data/splits/{0}/{1}/{4}/{2}{3}.csv'

    path = os.path.join(project_root,path_template.format(dataset, train_split, file_name_modifier, split, prediction_target))
    
    return pd.read_csv(path)


def save_score_label_df(df, 
                  dataset:str,
                  train_split:str, 
                  split: str,
                  path_sign: str,
                  file_name_modifier: str,
                  path_template = 'data/splits/{0}/{1}/{2}/{3}{4}.csv',
                  project_root= '/lotterlab/users/khoebel/xray_generalization'
                 ):
    
    # save score model labels
    # dataset: prediction dataset ('mmc', 'cxp')
    # train_split: percentage of data the model has been trained on 
    # model: name of the model used to generate the predictions 
    # split: dataset split ('train', 'val','test')
    # path_sign: label for which the score labels have been generated 
    # file_name_modifier: to comply with naming conventions between cxp and mmc ('', 'meta_')
    
    path = os.path.join(project_root, path_template.format(dataset, train_split, path_sign, file_name_modifier, split))
    df.to_csv(path)
        