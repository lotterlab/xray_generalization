### splits the training dataset into a new smaller training dataset and additional test data set
# (in addition to the conventional test dataset)
# split on a subject level 

import numpy as np 
import os
import pandas as pd
import random

from shutil import copy2

def create_subject_split(subjects, train_percentage):
    random.shuffle(subjects)
    split_ind = int(len(subjects)*train_percentage)
    train_inds = subjects[:split_ind]
    test_inds = subjects[split_ind:]
    # print('ind overlap', len(set(train_inds).intersection(test_inds)))

    return train_inds, test_inds


def split(train_percentage:float, 
          train_path_dict:dict, 
          dataset = 'cxp', # 'cxp', 'mimic'
          save_dir = '/lotterlab/users/khoebel/xray_generalization/data',
          seed = 0):
    
    random.seed(seed)

   
    if dataset == 'cxp':
        train_path = train_path_dict[dataset]
    elif dataset == 'mimic':
        train_path = train_path_dict[dataset][0]


    subject_name_dict = {'cxp': 'Patient', 
                         'mimic': 'subject_id'
                         }
    subject_col_name = subject_name_dict[dataset]

    train_df = pd.read_csv(train_path)
    print(train_df.head())
    subjects = train_df[subject_col_name].to_list()
    subjects = np.unique(subjects)
    print(len(subjects))

    quality_criterion_fulfilled = False
    count = 0
    while not quality_criterion_fulfilled:

        # create train/test splits and new dataframes 
        dataset_percentage = train_percentage/0.7 # the training df only contains 70% of the total labelled dataset
        train_ids, test_ids = create_subject_split(subjects, dataset_percentage)
        new_train_df = train_df.loc[train_df[subject_col_name].isin(train_ids)]
        new_test_df = train_df.loc[train_df[subject_col_name].isin(test_ids)]

        # test whether there is an overlap between train and test 
        assert len(set(new_train_df[subject_col_name]).intersection(set(new_test_df[subject_col_name]))) == 0
        
        # because I did the split on a subject level, 
        # I need to test whether there are roughly the desired number of studies in each new train/test split
        # 10% discrepancies are accepted 
        emp_size_train = new_train_df.shape[0]/train_df.shape[0]
        # print(emp_size_train)
        if (0.9*dataset_percentage)<= emp_size_train <= (1.1*dataset_percentage):
            quality_criterion_fulfilled = True
            print('dataset split quality criterion fulfilled', emp_size_train)
        count += 1 
        print('iteration', count)

    save_dir = os.path.join(save_dir, 'splits',dataset, str(train_percentage))
    os.makedirs(save_dir, exist_ok=True)
    if dataset == 'cxp':
        new_train_df.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
        new_test_df.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

        # copy the validation and original test splits
        val_path=train_path.replace('train', 'val')
        copy2(val_path, os.path.join(save_dir, 'val.csv'))

        test_path=train_path.replace('train', 'test')
        copy2(test_path, os.path.join(save_dir, 'test_orig.csv'))

    elif dataset == 'mimic':
        new_train_df.to_csv(os.path.join(save_dir, 'cxp-labels_train.csv'), index=False)
        new_test_df.to_csv(os.path.join(save_dir, 'cxp-labels_test.csv'), index=False)
        
        # copy the validation and original test splits
        val_path=train_path.replace('train', 'val')
        copy2(val_path, os.path.join(save_dir, 'cxp-labels_val.csv'))

        print(train_path)
        test_path=train_path.replace('train', 'test')
        print('test path', test_path)
        copy2(test_path, os.path.join(save_dir, 'cxp-labels_test_orig.csv'))
        


    

    if dataset == 'mimic':
        # apply split to metadata spreadsheet as well
        meta_df = pd.read_csv(train_path_dict[dataset][1])
        meta_train_df = meta_df.loc[meta_df['subject_id'].isin(train_ids)]
        meta_test_df = meta_df.loc[meta_df['subject_id'].isin(test_ids)]

        meta_train_df.to_csv(os.path.join(save_dir, 'meta_train.csv'), index=False)
        meta_test_df.to_csv(os.path.join(save_dir, 'meta_test.csv'), index=False)

        # copy validation and original test data
        # shutil.copy2(src_file, dest_file - need to rename the test data file ...
        val_path=train_path_dict[dataset][1].replace('train', 'val')
        copy2(val_path, os.path.join(save_dir, 'meta_val.csv'))
        test_path=train_path_dict[dataset][1].replace('train', 'test')
        copy2(test_path, os.path.join(save_dir, 'meta_test_orig.csv'))




if __name__ == "__main__":

    train_path_dict = {'cxp': '/lotterlab/lotterb/project_data/bias_interpretability/cxp_cv_splits/version_0/train.csv',
                       'mimic': ['/lotterlab/lotterb/project_data/bias_interpretability/mimic_cv_splits/version_0/cxp-labels_train.csv',
                                 '/lotterlab/lotterb/project_data/bias_interpretability/mimic_cv_splits/version_0/meta_train.csv']}

    for dataset in ['mimic', 'cxp']:
        split(.5,
            train_path_dict, 
            dataset = dataset)

