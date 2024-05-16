# Image-Based Predictive Trends for Chest X-Ray Classification

This repository contains the analysis code for the paper **"Beyond Structured Attributes: Image-Based Predictive Trends for Chest X-Ray Classification"**, which will be presented at MIDL 2024 in Paris. The code supports the experiments and analyses described in the paper, focusing on the systematic differences in prediction tendencies of models trained on different chest X-ray datasets.

## Paper
You can find the paper [here](link to paper).

## Model Training
All models (Pathology Prediction Models (PPMs) and Comparative Dataset Models (CDMs)) are trained using a modified version of the TorchXRayVision library, which can be found [here](https://github.com/lotterlab/torchxrayvision/tree/generalization).

### Training PPMs
To train a pathology prediction model on the MIMIC-CXR (MMC) dataset, use the following command:

```bash
python train_model.py --name model_name --dataset mimic_ch --model densenet --im_size 224 --fixed_splits --threads 12 --num_epochs 50 --all_views --imagenet_pretrained --use_no_finding
