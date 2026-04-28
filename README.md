# DLPromoter-SF:A Deep Learning-Based Approach of Predicting Saccharomyces cerevisiae Promoter Strength by Integrating Biological Statistical Features

## Overview

```DLPromoter-SF``` is a deep learning approach that combines sequence and statistical features to predict promoter strength in Saccharomyces cerevisiae. The method utilizes both one-hot encoding and statistical feature encoding as inputs to extract sequence information, subsequently employing a three-stage training process and ensemble learning for final prediction.

```DLPromoter-SF``` consists of three modules: the Sequence Feature Extraction Module,  Statistical Feature Extraction Module,  Feature Fusion and Output Module.The Sequence Feature Extraction module utilizes a multi-scale Convolutional Neural Network (CNN) integrated with Squeeze-and-Excitation (SE) attention mechanisms, alongside a Transformer encoder also featuring SE attention, to capture sequence-level information. The Statistical Feature Extraction module leverages FeedForward networks (FFN) and a gating mechanism to process statistical data. Finally, the Fusion and Output module employs Feature-wise Linear Modulation (FiLM) to facilitate effective feature integration, using a Multi-Layer Perceptron (MLP) to generate the predicted promoter strength.

The overall framework of ```DLPromoter-SF``` is shown in the following figure.

![DLPromoter-SF Model Architecture](https://github.com/wuwuwu12310/DLPromoter-SF/blob/main/Model_Architecture_.png)


## Description

The project includes the following core files and directory structure:
- The folder `dataset` contains the directory for storing the raw reaction data utilized in the training, testing andof DLPromoter-SF.
- The file `model.py` contains the definition of the multi-modal fusion model for DLPromoter-SF.
- The file `train.py` contains the script for training the DLPromoter-SF model.
- The file `test.py` contains the script for testing the DLPromoter-SF model.
- The file `best_model.pth`、 `best_S1_full.pth`、`best_S2_tail_head.pth` and `best_S3_unfreeze_tiny.pth`contains the script for the best model pth of DLPromoter-SF.
- The file `extra_norm.json` and `run_config.json` contains the script for the best model training configurations of DLPromoter-SF.

## System Requirements

The proposed ```DLPromoter-SF``` has been implemented, trained, and tested by using `Python 3.8` and `PyTorch 2.4.1` with `CUDA 12.1` and an `NVIDIA RTX4090` graphics card.

The package depends on the Python scientific stack:
```
Python： 3.8.20
torch： 2.4.1
Transformers：  4.46.3 
scikit-learn ： 1.3.2
pandas:  1.5.0 
numpy : 1.23.5 
tqdm:  4.46.1
scipy:  1.9.3  
```

## Usage

### Datasets

The dataset of a large-scale dataset comprising 162,982 80-bp Saccharomyces cerevisiae promoter sequences and their respective strength is available [here](https://github.com/RK2627/PromoDGDE/tree/main/Data/SC).
The dataset of a small-scale dataset comprising 63,468 80-bp Saccharomyces cerevisiae promoter sequences and their respective strength is available [here](https://github.com/1edv/evolution/blob/master/manuscript_code/model/reproduce_test_data_performance).


### Model Training
We define the multimodal model for the **DLPromoter-SF** method in the file `model.py`, where:
- The sequence feature extraction module is implemented based on the multi-scale Convolutional Neural Network (CNN) integrated with Squeeze-and-Excitation (SE) attention mechanisms and Transformer encoders with Squeeze-and-Excitation (SE) attention mechanisms.
- The statistical feature extraction module is implemented based on the FeedForward networks (FFN) and a gating mechanism to process statistical data.
The model can be trained using the file `train.py`.

### Model testing
- Run `test.py` using the new .pth and new .pth model weights:run_config.json, and extra_norm.json obtained from the `train.py` output.



