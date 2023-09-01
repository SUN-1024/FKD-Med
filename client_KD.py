import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from DataSet import segDataset
import torchvision
from torchvision import transforms
import os

import glob
from PIL import Image
from datetime import datetime
import argparse


from model.simpleUnet import simpleUnet as simpleUnet
from model.Unet_model.Unetpp import NestedUNet as Unetpp
from model.Unet_model.AttentionUnet import AttU_Net as AttUnet

from model.Unet_model.ResUnet.res_unet import ResUnet as ResUnet
from model.VggUnet import UNet as mldaUnet

from model.transUnet.vit_seg_modeling import VisionTransformer as transUnet
from model.transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
DEVICE2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# T = 5 is best
T = 2
# alpha = 0.8 is best, 0.9 is too much, less than 0.8 decreases accuracy
# alpha = 0.8
alpha = 0.5

def pixel_wise_loss(student_output, teacher_output):
    _,C,W,H = student_output.shape
  
    pred_T = torch.sigmoid(teacher_output/T)
    pred_S = torch.sigmoid(student_output/T).log()
  
    #TODO: map this to KLDL
  
    pixelwise_loss = (- pred_T * pred_S)
  
    return  torch.sum(pixelwise_loss) / (W*H*10)



def loss_fn(student_output, teacher_output, gt , criterion):
    '''student_output = student_output.round()
    student_output[student_output<0] = 0
    gt = torch.clamp(gt, min = 0, max = 1)
    teacher_output = torch.clamp(teacher_output, min = 0, max = 1)'''

    student_output = student_output.clamp(min = 0, max = 1)
    teacher_output = teacher_output.clamp(min = 0, max = 1)
    student_loss = criterion(student_output, gt)
    kd_loss = pixel_wise_loss(student_output, teacher_output)
    loss = (student_loss*(1-alpha) + (kd_loss)*(alpha)) # as per structured KD paper
    return loss


def train(net,teacher, trainloader, epochs):
    """Train the model on the training set."""

    for i in range(epochs):
        print("\n"+'The',i+1,' time is training')

        correct = 0
        total = 0
        running_loss = 0
        teacher.eval()
        net.train()
        for images, labels in tqdm(trainloader):
            # if (mps == True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                teach_pred  = teacher(images.to(DEVICE))
            labels_pred = net(images)

            loss = loss_fn(labels_pred, teach_pred, labels.to(DEVICE), criterion)

            params = list(net.parameters())
            for param in params:
                print(param.shape)  # 打印每个参数的形状
            print("....")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                labels_pred = torch.argmax(labels_pred, dim=1)
                correct += (labels_pred == labels).sum().item()
                total += labels_pred.size(0)
                running_loss += loss.item()

        exp_lr_scheduler.step()
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct / (total*256*256)
        print('the ',i,' time')
        print('loss:',epoch_loss)
        print('acc:',epoch_acc)
        losslist.append(epoch_loss)

def test(net, testloader):
    """Validate the model on the test set."""
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    for images, labels in tqdm(testloader):
        with torch.no_grad():
            # if (mps == True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            teach_pred = teacher(images.to(DEVICE))
            outputs = net(images.to(DEVICE))
            loss = criterion(outputs,labels)

            outputs = torch.argmax(outputs, dim=1)
            # labels = labels.to(DEVICE)
            test_correct += (outputs == labels).sum().item()
            test_total += labels.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total*256*256)
    print('test_loss： ', epoch_test_loss,
          'test_accuracy:',epoch_test_acc
          )
    plot_examples(images,labels,outputs)
    return epoch_test_loss,epoch_test_acc


def load_data(client):
    t = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor(),])


    imgs_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/images/*')
    imgs_path.sort()
    labels_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/masks/0/*')
    labels_path.sort()




    if client == "train":
        i = int(len(imgs_path)*0.8)
        train_path = imgs_path[ :i]
        train_labels_path = labels_path[ :i]
        test_path = imgs_path[i: ]
        test_labels_path = labels_path[i: ]
        BACH_SIZE=4
        TrainDataSet = segDataset(img_paths = train_path,anno_paths = train_labels_path,transform = t)
        TestDataSet  = segDataset(img_paths = test_path,anno_paths = test_labels_path,transform = t)
        trainset = data.DataLoader(TrainDataSet,batch_size=BACH_SIZE,
                                   shuffle=True, num_workers=0)#num_works =8
        testset = data.DataLoader( TestDataSet, batch_size=BACH_SIZE,
                                   shuffle=False, num_workers=0)#num_works =8
    elif client == "client1":
        i = int(len(imgs_path)*0.5)
        i1 =int(len(imgs_path)*0.9)
        train_path = imgs_path[ :i]
        train_labels_path = labels_path[ :i]
        test_path = imgs_path[i1:]
        test_labels_path = labels_path[i1:]
        BACH_SIZE=4
        TrainDataSet = segDataset(img_paths = train_path,anno_paths = train_labels_path,transform = t)
        TestDataSet  = segDataset(img_paths = test_path,anno_paths = test_labels_path,transform = t)
        trainset = data.DataLoader(TrainDataSet,batch_size=BACH_SIZE,
                                   shuffle=True, num_workers=0)#num_works =8
        testset = data.DataLoader( TestDataSet, batch_size=BACH_SIZE,
                                   shuffle=False, num_workers=0)#num_works =8
    elif client == "client2":
        i = int(len(imgs_path)*0.5)
        i1 =int(len(imgs_path)*0.9)
        train_path = imgs_path[i:i1]
        train_labels_path = labels_path[i:i1]
        test_path = imgs_path[i1: ]
        test_labels_path = labels_path[i1: ]
        BACH_SIZE=4
        TrainDataSet = segDataset(img_paths = train_path,anno_paths = train_labels_path,transform = t)
        TestDataSet  = segDataset(img_paths = test_path,anno_paths = test_labels_path,transform = t)
        trainset = data.DataLoader(TrainDataSet,batch_size=BACH_SIZE,
                                   shuffle=True, num_workers=0)#num_works =8
        testset = data.DataLoader( TestDataSet, batch_size=BACH_SIZE,
                                   shuffle=False, num_workers=0)#num_works =8
    print('-----------------Load Data---------------------------')
    return trainset, testset

def saveData(new_data):
    with open('DataList/dataTestFKD.txt', 'a') as file:
        # 写入新数据
        for num in new_data:
            file.write(str(num) + '\n')

def plot_examples(image, label, pre):
    pre = pre.float()
    label = label.float()

    original_tensor = image[0]
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    pic.save('picFKD/imageFKD.jpg')

    original_tensor = label[0]
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    pic.save('picFKD/labelFKD.jpg')

    original_tensor = pre[0]
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    pic.save('picFKD/preFKD.jpg')

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, default = "",required=True, help="path to your train dataset")
    # #Train
    # parser.add_argument("--test", type=str, default = "",help="path to your test dataset")
    #Test
    parser.add_argument("--client", type=str, default="client1", help="path to your data")
    parser.add_argument("--cuda", type=str, default="cuda2", help="path to your data")
    # parser.add_argument("--meta", type=str, required=True, help="path to your metadata")
    # parser.add_argument("--name", type=str, default="unet", help="name to be appended to checkpoints")
    # parser.add_argument("--DataSet", type=str, default="unet", help="name to be appended to checkpoints")
    # parser.add_argument("--model", type=str, default="simpleUnet", help="AttentionUnet | DAUnet | VggUnet | simpleUnet")
    # parser.add_argument("--loss",type=str,default="crossentropy",
    #                     help="focalloss | iouloss | crossentropy",
    #                     )#Loss
    #
    parser.add_argument("--num_epochs", type=int, default=10, help="dnumber of epochs")
    # #Epochs
    # parser.add_argument("--batch", type=int, default=1, help="batch size")
    # #batch_size
    # parser.add_argument("--save_step", type=int, default=5, help="epochs to skip")

    return parser.parse_args()

if __name__ == "__main__":
    StartTime = datetime.now()
    args = get_args()
    epochs = args.num_epochs
    client = args.client
    cuda = args.cuda
    if cuda == "cuda2":
        DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:3")
    elif cuda == "cuda3":
        DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cuda:2")
    # # AttentionUnet | DAUnet | VggUnet | simpleUnet
    # if args.model == "AttentionUnet":
    #     net = AttentionUnet(num_classes=2,num_channels=1).to(DEVICE)
    # elif args.model == "DAUnet":
    #     net = DAUnet(num_classes=2,num_channels=1).to(DEVICE)
    # elif args.model == "VggUnet":
    #     net = VggUnet(num_classes=2,num_channels=1).to(DEVICE)
    # elif args.model == "simpleUnet":
    #     net = simpleUnet(num_classes=2,num_channels=1).to(DEVICE)

    modelStu = simpleUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelStu = mldaUnet(num_classes=2,num_channels=1).to(DEVICE)

    # modelTea = simpleUnet(num_classes=2,num_channels=1).to(DEVICE)
    modelTea = AttUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = Unetpp(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = MulUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = ResUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = mldaUnet(num_classes=2,num_channels=1).to(DEVICE)
    net = modelStu
    teacher = modelTea

    if client == "client1":
        PATH = 'saved_models/client1_model.pth'
    elif client == "client2":
        PATH = 'saved_models/client2_model.pth'
    teacher.load_state_dict(torch.load(PATH))#????

    trainloader , testloader = load_data(client)
    
    # net = nn.DataParallel(net)
    # model = UNet(n_channels=3, n_classes=n_classes, bilinear=True, channel_depth=8).to(device)
    # teacher = UNet(n_channels=3, n_classes=n_classes, bilinear=True, channel_depth=64).to(device)
    #
    # teacher.load_state_dict(torch.load('ENTER SAVED MODEL WEIGHT HERE'))
    # 这是保存不是读取
    # is Save not Get
    # my_model = net()
    # my_model.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()#loss_fn
    
    from torch.optim import lr_scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # where be used?
    # Start Flower client
    print('get all par,it is begin')
    losslist=[]
    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            print('\nget parameters')
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):#get the value from Server
            print('\nset parameters and FedAvg')
            global net
            params_dict = zip(net.state_dict().keys(), parameters)
            # local variable 'net' referenced before assignment
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # net = nn.DataParallel(net).cuda()
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            print("\nTraining started on Client...")
            self.set_parameters(parameters)
            train(net,teacher,trainloader, epochs)
            return self.get_parameters(config={}), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            print("\nEvaluation started on Client...")
            net.eval()
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            print(loss,accuracy)
            # losslist.append(loss)
            return loss, len(testloader.dataset), {"accuracy": accuracy}

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )
    EndTime = datetime.now()
    print(losslist)
    print ("StartTime is %s" % StartTime)
    print("EndTime is %s" % EndTime)
    UseTime = (EndTime - StartTime)
    print("UseTime is %s" % UseTime)
    