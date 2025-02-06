# CHANGE-net: Deep learning framework for CRISPR off-target activity prediction

## Overview

**CHANGE-net** is a deep learning framework for predicting off-target editing activity. It leverages convolutional neural network (CNN) to extract information from one-hot encoded DNA sequence and identify the important sequence features. It is trained using CHANGE-seq-R data and applied on real genetic variants.

This repository contains the data preprocessing, model training, and model inference code. It also includes the pre-trained model weights and a small toy dataset for inference.

## Installation
### Requirements

CHANGE-net has been developed and tested in the following environment:  

- python==3.9
- h5py==3.10.0
- matplotlib==3.8.2
- numpy==1.26.4
- pandas==2.2.2
- scikit_learn==1.4.0
- scipy==1.13.1
- seaborn==0.13.2
- torch==1.10.1
- tqdm==4.66.1
- notebook==7.1.3
- jupyterlab==4.1.6

You can install the dependencies using:  

```bash
conda create -n changenet python==3.9
conda activate changenet
pip install -r requirements.txt
```
Typical installation time on a normal desktop computer: **~5 minutes**  

## Demo  

The following instructions demonstrate how to run the code using the toy dataset.  

### 1. Prepare Data  

The input is a CSV-formatted table generated from the CHANGE-seq-R workflow.  
To prepare the data for the machine learning framework, run `data_processing.ipynb`.  
This step will generate hdf5 files for each target.

### 2. Train the Model  

To train CHANGE-net, follow the notebook `model_training_evaluation.ipynb`.  
In this demo, we skip the training step since a small dataset may not provide optimal performance. Instead, we provide pre-trained model weights.  

### 3. Run inference

To perform inference on new data, use the provided pre-trained model weights and follow the `model_inference.ipynb` notebook.  
Inference on the toy dataset should take only a few seconds on a single CPU.
The expected output file is included in the `results/` directory.

## Instructions for Use  

### Running the model on your data  

1. Prepare your data in the same format as the provided toy dataset, with the minimum required columns: `site`, `MM`, and `control_target_seq`.  
2. Modify paths in `data_processing.ipynb` and `inference.ipynb` to point to your dataset.  
3. Run the processing and inference notebooks to obtain predictions.  