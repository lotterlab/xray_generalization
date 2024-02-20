import pdb
import skimage
import pandas as pd
import numpy as np
import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
import os
import copy
import tqdm

import sys
sys.path.append('../torchxrayvision/')
import torchxrayvision as xrv
sys.path.append('../pytorch-grad-cam/')
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import read_prediction_df
from pycrumbs import tracked

def perform_xrv_preprocessing(im_path):
    img = skimage.io.imread(im_path)
    img = xrv.datasets.normalize(img, maxval=255, reshape=True)
    transforms = [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
    transform = torchvision.transforms.Compose(transforms)
    img = transform(img)
    return img


def apply_colormap_on_image(org_im, activation, colormap_name, threshold=0.3):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
        threshold (float): threshold at which to overlay heatmap

    Original source: https://github.com/utkuozbulak/pytorch-cnn-visualizations

    Added thresholding to activations.
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4

    # set to fully transparent if there is a very low activation
    idx = (activation <= threshold)
    # convert to a 3d index
    ignore_idx = np.expand_dims(np.zeros(activation.shape, dtype=bool), 2)
    idx = np.concatenate([ignore_idx] * 3 + [np.expand_dims(idx, 2)], axis=2)

    heatmap[idx] = 0

    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image


def generate_cam_image(cam_model, f_path, target_num):
    orig_img = perform_xrv_preprocessing(f_path)

    input_tensor = torch.from_numpy(orig_img).unsqueeze(0)
    targets = [ClassifierOutputTarget(target_num)]

    grayscale_cam = cam_model(input_tensor=input_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    img = input_tensor.cpu().numpy()[0, 0, :, :]
    img = (img / 1024.0 / 2.0) + 0.5
    img = np.clip(img, 0, 1)
    img = Image.fromarray(np.uint8(img * 255), 'L')

    cam_on_image = apply_colormap_on_image(img, grayscale_cam, 'cool', threshold=0.4)

    return grayscale_cam, cam_on_image


def load_cam_model(orig_model_path):
    this_model = torch.load(orig_model_path)
    this_model = this_model.eval()
    target_layer = [this_model.features[-1]]
    cam_model = GradCAM(model=this_model, target_layers=target_layer, use_cuda=True)
    return cam_model


# TODO: add some config of who ran etc (pycrumbs)
# @tracked(directory_parameter='record_dir')
def run_cam_generation(model_name, model_path, train_split, cam_dataset, prediction_type,
                       pred_label, prediction_target, # pathology or pneumonia/pneumothorax etc for score models
                       cam_split='val', n_cams=50,
                       record_dir=None):

    cam_model = load_cam_model(model_path)

    # just hard code some stuff for now
    '''if prediction_type == 'pathology':
        prediction_target = 'pathology'
    elif prediction_type == 'higher_score':
        prediction_target = 'pneumothorax'
        '''

    # get files to predict
    pred_df = read_prediction_df(cam_dataset, train_split, model_name, cam_split, prediction_target, merge_labels=True)

    pred_df.sort_values('Pred_' + pred_label, ascending=False, inplace=True)
    cam_files = pred_df.Path.values[:n_cams]

    # TODO: double check that these are all correct
    if pred_label == 'CXP':
        target_idx = 0
    elif pred_label == 'MMC':
        target_idx = 1
    else: #pnx should be 8
        pdb.set_trace()

    # generate and save cams
    base_out_dir = f'/lotterlab/project_data/cxr_generalization/grad_cams/{prediction_target}/'
    model_folder = f'{model_name}_{train_split}'
    cam_folder = f'camdata-{cam_dataset}-{cam_split}_pred-{pred_label}'
    out_dir = os.path.join(base_out_dir, model_folder, cam_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(out_dir)

    for i, f_path in tqdm.tqdm(enumerate(cam_files), total=len(cam_files)):
        grayscale_cam, cam_on_image = generate_cam_image(cam_model, f_path, target_idx)
        # save image
        cam_on_image.save(os.path.join(out_dir, f'{i}.png'))

        # normalize cam
        grayscale_cam = grayscale_cam / grayscale_cam.max()
        
        if i == 0:
            av_cam = grayscale_cam
        else:
            av_cam = (i / (i + 1)) * av_cam + (1 / (i + 1)) * grayscale_cam

    # save average cam
    np.save(os.path.join(out_dir, 'average_cam.npy'), av_cam)

    norm_av_cam = (av_cam * 255).astype(np.uint8)
    norm_image = Image.fromarray(norm_av_cam, mode='L')
    norm_image.save(os.path.join(out_dir, 'average_cam.png'))

    norm_av_cam = (av_cam - av_cam.min()) / (av_cam.max() - av_cam.min())
    norm_av_cam = (norm_av_cam * 255).astype(np.uint8)
    norm_image = Image.fromarray(norm_av_cam, mode='L')
    norm_image.save(os.path.join(out_dir, 'average_cam-norm01.png'))


if __name__ == '__main__':
    
    # model_path = '/lotterlab/users/khoebel/xray_generalization/models/mmc/0.7/pneumothorax/mmc_score_0.7_seed_1/mimic_ch-densenet-mmc_score_0.7_seed_1-best.pt'
    train_split = 0.7
    # cam_dataset = 'mmc'
    prediction_type = 'higher_score'
    # pred_label = 'MMC'
    cam_split = 'val'
    prediction_target = 'effusion'
    model_path_dict = {'mmc': f'/lotterlab/users/khoebel/xray_generalization/models/mmc/0.7/{prediction_target}/mmc_score_0.7_seed_1/mimic_ch-densenet-mmc_score_0.7_seed_1-best.pt',
                       'cxp': f'/lotterlab/users/khoebel/xray_generalization/models/cxp/0.7/{prediction_target}/cxp_score_0.7_seed_1/chex-densenet-cxp_score_0.7_seed_1-best.pt'
    }
    model_name_dict = {'mmc':'mmc_score_0.7_seed_1-best', 
                       'cxp':'cxp_score_0.7_seed_1-best'
    }
    for cam_dataset in ['cxp', 'mmc']:
        model_path = model_path_dict[cam_dataset]
        model_name = model_name_dict[cam_dataset]
        # ToDo: move inside function (for pycrumbs documentation)
        base_out_dir = '/lotterlab/project_data/cxr_generalization/grad_cams/'
        model_folder = f'{model_name}_{train_split}'
        
        for pred_label in ['CXP', 'MMC']:
            cam_folder = f'camdata-{cam_dataset}-{cam_split}_pred-{pred_label}'
            record_dir = os.path.join(base_out_dir,prediction_target, model_folder, cam_folder)
            if not os.path.exists:
                os.makedirs(record_dir)
            run_cam_generation(model_name, model_path, train_split, cam_dataset, prediction_type,
                            pred_label, prediction_target,cam_split=cam_split, 
                            record_dir = record_dir)