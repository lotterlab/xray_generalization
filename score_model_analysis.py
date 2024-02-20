import pdb
from functools import partial
import pandas as pd
from PIL import Image
import numpy as np
import torchvision as tv
import torch
import tqdm
import matplotlib.pyplot as plt
import skimage
import os
import sys
import torch.nn.functional as F

from generate_gradcams import perform_xrv_preprocessing
from utils import read_prediction_df, read_dataset_df
sys.path.append('../torchxrayvision/')
import torchxrayvision as xrv


def compute_mean_image(file_paths, histogram=True, hist_range=(-1024,1024), n_bins=100):
    av_im = None
    count = 0
    
    if histogram:
        av_hist = None
        
    for f in tqdm.tqdm(file_paths):
        try:
            img = perform_xrv_preprocessing(f)
            if av_im is None:
                av_im = img
            av_im = (av_im * count + img) / (count + 1)
            count += 1
            
            if histogram: 
                if av_hist is None:
                    av_hist = np.asarray(np.histogram(img, bins=n_bins, range=hist_range))
                else:
                    av_hist[0] = av_hist[0]+np.histogram(img, bins=n_bins, range=hist_range)[0]
        except:
            pass
    if histogram:
        av_hist[0] = av_hist[0]/count
        return av_im[0,:,:], av_hist
    else:
        return av_im[0,:,:]
    

def compute_center_vals(bins):
    center = np.asarray([(bins[i]+bins[i+1])/ 2 for i in range(len(bins)-1)])
    return center


def compute_feature_frequency(sorted_subgroup_df, feature, n_bins):
    feature_values = [x for x in sorted_subgroup_df[feature].unique() if str(x)!='nan']
    # print(feature_values)
    
    # create dict to store frequency values 
    value_frequencies = {k: list() for k in feature_values}
    
    # loop through the subgroup dataframe and collect feature frequencies 
    bin_width = int(np.floor(sorted_subgroup_df.shape[0]/n_bins))
    
    for i in range(n_bins):
        if i != n_bins-1:
            temp_df = sorted_subgroup_df[i*bin_width:(i+1)*bin_width]
        else: 
            temp_df = sorted_subgroup_df[i*bin_width:]
            bin_width = temp_df.shape[0]
        
        # print(np.mean(temp_df['Pred_CXP'].values), bin_width)
        temp_counts = temp_df[feature].value_counts()
        for k, value_list in value_frequencies.items():
            try:
                value_list.append(temp_counts[k]/bin_width)
            except:
                value_list.append(0)

    return value_frequencies

def compute_value_frequencies_wrapper(dataset, model_name, train_split, split, 
                                      feature_col, n_bins = 10,
                                      score_model_target='pneumothorax',
                                      project_root='/lotterlab/users/khoebel/xray_generalization'):
    
    # load meta data 
    if dataset == 'mmc':
        file_name_modifier = 'meta_'
    elif dataset == 'cxp':
        file_name_modifier = ''
    else:
        raise ValueError(f"Invalid value for mode: {dataset}. Mode must be 'cxp' or 'mmc'.")
    
    # TODO: should be updated once we have the complete subgroup analysis dataframes 
    meta_data = read_dataset_df(dataset=dataset,
                    train_split=train_split,
                    file_name_modifier=file_name_modifier,
                    split=split,
                    prediction_target='pathology',
                    path_template = None,
                    project_root = project_root
                   )
    
    # load score model predictions
    pred_df = read_prediction_df(dataset = dataset,
                     train_split = train_split,
                     model = model_name,
                     split = split,
                     prediction_target = score_model_target,
                     merge_labels = True)
    
    # merge meta data and predictions 
    # define columns for datset merger
    if dataset == 'cxp':
        left_col = 'Path'
        right_col = 'orig_path'
    elif dataset == 'mmc':
        left_col = 'dicom_id'
        right_col = 'dicom_id'
    subgroup_df = meta_data.merge(pred_df, left_on=left_col, right_on=right_col, how='inner')
    # sort by CXP prediction and reset index 
    subgroup_df.sort_values('Pred_CXP', ascending=False, inplace=True)
    subgroup_df = subgroup_df.reset_index()

    # compute frequencies
    feature_frequencies = compute_feature_frequency(subgroup_df, feature_col, n_bins=n_bins)

    return feature_frequencies


def plot_frequencies(value_frequencies, title=None):
    for k, v in value_frequencies.items():
        plt.plot(v, label=k)
    
    plt.legend()
    plt.xlabel('CXP score pred [percentile bins]')
    plt.ylabel('feature frequency')
    if title is not None:
        plt.title(title)

