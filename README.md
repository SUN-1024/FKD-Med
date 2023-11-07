# FKD-Med  

FKD-Med: Privacy-Aware, Communication-Optimized Medical Image Segmentation via Federated Learning and Model Lightweighting through Knowledge Distillation

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


# FKD-Med Framework

## Overview
This repository contains code for training and testing U-Net for image semantic segmentation tasks. It contains both traditional and Federated Traning using the FedAvg algorithm in the Flower framework.
_All operations of the user are done in this directory_

## Getting Datastes
### CVC-ClinicDB Datasets
https://paperswithcode.com/dataset/cvc-clinicdb

### Chest Xray Masks and Labels Datasets
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
## Framework

### Model
In the `/model/Unet_model`,training on different Unet models,


### Loss
In the `Loss.py`,choosing different loss functions for training

### Dataset

In the `DataSet.py`,selecting different medical data for training


## Unified Unet Commands(CVC-ClinicDB)
The model is customized in the command line for the reader to choose from
```sh
python3 train.py --client client1 --cuda cuda1 --model resUnet --num_epochs 50 --dataset CVC --picFormat .tif
```
```sh
python3 train.py --client client2 --cuda cuda2 --model resUnet --num_epochs 50  --dataset CVC --picFormat .tif
```
```sh
python3 train.py --client client3 --cuda cuda3 --model resUnet --num_epochs 50  --dataset CVC --picFormat .tif
```

## Unified Unet Commands(Chest Xray)
The model is customized in the command line for the reader to choose from
```sh
python3 train.py --client client1 --cuda cuda1 --model resUnet --num_epochs 50 --dataset Chest --picFormat .png
```
```sh
python3 train.py --client client2 --cuda cuda2 --model resUnet --num_epochs 50  --dataset Chest --picFormat .png
```
```sh
python3 train.py --client client3 --cuda cuda3 --model resUnet --num_epochs 50  --dataset Chest --picFormat .png
```

## Federated Train Commands(CVC-ClinicDB)

```sh
python3 server.py
```
```sh
python3 client.py --client client1 --cuda cuda1 --model resUnet --num_epochs 10 --dataset CVC --picFormat .tif
```
```sh
python3 client.py --client client2 --cuda cuda2 --model resUnet --num_epochs 10 --dataset CVC --picFormat .tif
```
```sh
python3 client.py --client client3 --cuda cuda3 --model resUnet --num_epochs 10 --dataset CVC --picFormat .tif
```

## Federated Train Commands(Chest Xray)

```sh
python3 server.py
```
```sh
python3 client.py --client client1 --cuda cuda1 --model resUnet --num_epochs 10 --dataset Chest --picFormat .png
```
```sh
python3 client.py --client client2 --cuda cuda2 --model resUnet --num_epochs 10 --dataset Chest --picFormat .png
```
```sh
python3 client.py --client client3 --cuda cuda3 --model resUnet --num_epochs 10 --dataset Chest --picFormat .png
```

## FedKD Train Commands

### Teacher Modeling Training Prior to Federal Learning
The teacher model has been trained in the `train.py` file according to client,line 202,weights are found inside `saved_models/` folder Eg.
```sh
torch.save(model.state_dict(),PATH)
```
The trained teacher model was used at line 129 of `clientFKD.py`
```sh
teacher.load_state_dict(torch.load(PATH))
```
## Student Model Commands(CVC-ClinicDB)
```sh
python3 server.py
```
```sh
python3 clientFKD.py --client client1 --cuda cuda1 --model resUnet --num_epochs 10 --dataset CVC --picFormat .tif
```
```sh
python3 clientFKD.py --client client2 --cuda cuda2 --model resUnet --num_epochs 10 --dataset CVC --picFormat .tif
```
```sh
python3 clientFKD.py --client client3 --cuda cuda3 --model resUnet --num_epochs 10 --dataset CVC --picFormat .tif
```

## Student Model Commands(Chest Xray)
```sh
python3 server.py
```
```sh
python3 clientFKD.py --client client1 --cuda cuda1 --model resUnet --num_epochs 10 --dataset Chest --picFormat .png
```
```sh
python3 clientFKD.py --client client2 --cuda cuda2 --model resUnet --num_epochs 10 --dataset Chest --picFormat .png
```
```sh
python3 clientFKD.py --client client3 --cuda cuda3 --model resUnet --num_epochs 10 --dataset Chest --picFormat .png
```

<!-- ## Review of results
### The test image shows
```sh
PicList
    ---pic
    ---picFed
    ---picFKD
```
### The test loss function shows
```sh
DataList
    ---Unet
    ---Fed
    ---FKD
``` -->
