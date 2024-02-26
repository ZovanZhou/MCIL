# MCIL

[Title] MCIL: Multimodal Counterfactual Instance Learning for Low-resource Entity-based Multimodal Information Extraction

## Preparation

1. Clone the repo to your local.
2. Download Python version: 3.6.13
3. Download the dataset from this [link](https://pan.baidu.com/s/1g4qda-y7SsPHZbxYElhyjQ ) and the extraction code is **1234**. Unzip the downloaded files into the ''**dataset**'' folder.
4. Download the [BERT](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) pretrained models, and unzip into the ''**pretrain**'' folder.
5. Open the shell or cmd in this repo folder. Run this command to install necessary packages.

```cmd
pip install -r requirements.txt
```

## Experiments

1. In each folder "MET" or "MRE", we have shell scripts to run the training procedures for the Linux systems. You can run the following command:

```cmd
./run.model.sh 0 1 umt
```

2. You can also input the following command to train the model. There are different choices for some hyper-parameters shown in square barckets. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|n_train|int|Number of labeled data samples|
|cf|int|Whether to use MCIL during training|
|model|string|Multimodal models including: umt, mkgformer|

```cmd
CUDA_VISIBLE_DEVICES=1 \
python main.py \
    --n_train 100 \
    --seed 0 \
    --cf 1 \
    --model umt
```
