# Image-Based Predictive Trends for Chest X-Ray Classification

This repository contains the analysis code for the paper **"Beyond Structured Attributes: Image-Based Predictive Trends for Chest X-Ray Classification"**, which will be presented at MIDL 2024 in Paris. The code supports the experiments and analyses described in the paper, focusing on the systematic differences in prediction tendencies of models trained on different chest X-ray datasets.

## Paper
You can find the paper [here](https://openreview.net/pdf?id=Y15taNvfFN).

## Model Training
All models (Pathology Prediction Models (PPMs) and Comparative Dataset Models (CDMs)) are trained using a modified version of the TorchXRayVision library, which can be found [here](https://github.com/lotterlab/torchxrayvision/tree/generalization).

### Training PPMs
To train a pathology prediction model on the MIMIC-CXR (MMC) dataset, use the following command:

```bash
python train_model.py --name model_name --dataset mimic_ch --model densenet --im_size 224 --fixed_splits --threads 12 --num_epochs 50 --all_views --imagenet_pretrained --use_no_finding
```


### Training CDMs
CDM training labels are generated based on the predictions of PPMs trained on the CheXpert (CXP) and MMC datasets using ```create_score_labels.py```.

To train a CDM using the CXP dataset, use the following command:
```bash
python train_model.py --name model_name --dataset chex --model densenet --im_size 224 --fixed_splits --fixed_splits_source path_to_CXP_predictive_tendency_labels --num_epochs 50 --all_views --imagenet_pretrained --label_type higher_score --no_taskweights
```
To train a CDM using the MMC dataset, use the following command:
```bash
python train_model.py --name model_name --dataset mimic_ch --model densenet --im_size 224 --fixed_splits --fixed_splits_source path_to_MMC_pathology_labels --fixed_splits_mmc_score_source path_to_MMC_predictive_tendency_labels --num_epochs 50 --all_views --imagenet_pretrained --label_type higher_score --no_taskweights
```


__Training on Modified Images__
To train models on modified images that selectively remove low and high-frequency content or randomize the positon of pixels within an image, use the following flags:

- High pass filter:
   ```--use_high_pass_filter filter_radius```
- Low pass filter:
   ```--use_low_pass_filter filter_radius```
- Randomize pixels:
   ```--randomize_pixels```

### Model Weights
The weights for trained PPMs and CDMs can be found [here](link to folder with weights).

## Analysis
To replicate part of the analysis presented in the paper, we provide a Jupyter notebook along with the required data files (CXP_Pneumothorax.csv and MMC_Pneumothorax.csv). These spreadsheets are needed for the example analysis illustrated in ```PneumothoraxAnalysis.ipynb```.

## Citation
If you use this code in your research, please cite our paper:

@inproceedings{hoebel2024image,
  title={Beyond Structured Attributes: Image-Based Predictive Trends for Chest X-Ray Classification},
  author={Hoebel, Katharina and Fernando, Jesseba and Lotter, William},
  booktitle={Proceedings of the Medical Imaging with Deep Learning (MIDL) Conference},
  year={2024}
}

