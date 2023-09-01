import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import os
import sys
import glob
from PIL import Image
from datetime import datetime
import argparse
from _UnifiedModule import *
from collections import OrderedDict
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# T = 5 is best
T = 5
alpha = 0.8

#     return loss

def loss_fn(student_output, teacher_output,labels):

    KD_loss = nn.KLDivLoss()(F.log_softmax(student_output/T, dim=1),
                             F.softmax(teacher_output/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(student_output, labels) * (1. - alpha)
    return KD_loss


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)

def plot_examples(image,label,pre,client):
    pre = pre.float()
    label = label.float()

    original_tensor = image[0]
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    pic.save('PicList/picFKD/imageFed.jpg')

    original_tensor = label[0]
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    pic.save('PicList/picFKD/labelFed.jpg')

    original_tensor = pre[0]
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    if client == "client1":
        pic.save('PicList/picFKD/pre1.jpg')
    elif client == "client2":
        pic.save('PicList/picFKD/pre2.jpg')
    elif client == "client3":
        pic.save('PicList/picFKD/pre3.jpg')
if __name__ == "__main__":
    num_plot=0
    StartTime = datetime.now()
    args = get_args()
    epochs = args.num_epochs
    client = args.client
    cuda = args.cuda
    dataset = args.dataset
    picFormat = args.picFormat
    filename = os.path.basename(sys.argv[0])[:-3]
    
    if cuda =="cuda1":
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:2")
    elif cuda == "cuda2":
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
    modelName = args.model
    if modelName == "simpleUnet":#1--3
        model = simpleUnet(num_classes=2,num_channels=1).to(DEVICE)
    elif modelName == "resUnet":
        model = ResUnet(num_classes=2,num_channels=1).to(DEVICE)
    elif modelName == "transUnet":
        # config_vit = CONFIGS_ViT_seg[args.vit_name]
        model = transUnet(CONFIGS_ViT_seg[args.vit_name],args.input, num_classes = 2).to(DEVICE)
    else:
        print("Not Find the Model!")
    modelTea = model
    # modelTea = simpleUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = AttUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = Unetpp(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = MulUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = ResUnet(num_classes=2,num_channels=1).to(DEVICE)
    # modelTea = mldaUnet(num_classes=2,num_channels=1).to(DEVICE)
    #______+TransformUnet
    net = modelStu
    teacher = modelTea

    # if client == "client1":
    #     PATH = 'saved_models/client1_model.pth'
    # elif client == "client2":
    #     PATH = 'saved_models/client2_model.pth'
    # elif client == "client3":
    #     PATH = 'saved_models/client3_model.pth'
    # else:
    #     print("load_state_dict is error!")
    PATH = os.path.join('saved_models',client+'_model.pth')
    teacher.load_state_dict(torch.load(PATH))

    trainloader , testloader = load_data(args=args)

    # net = nn.DataParallel(net)
    # model = UNet(n_channels=3, n_classes=n_classes, bilinear=True, channel_depth=8).to(device)
    # teacher = UNet(n_channels=3, n_classes=n_classes, bilinear=True, channel_depth=64).to(device)
    #
    # teacher.load_state_dict(torch.load('ENTER SAVED MODEL WEIGHT HERE'))
    # my_model = net()
    # my_model.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()#loss_fn
    # criterion = nn.BCEWithLogitsLoss()

    from torch.optim import lr_scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # where be used?
    # Start Flower client
    print('get all par,it is begin')
    losslist=[]
    def train(net,teacher,trainloader, epochs):
        """Train the model on the training set."""
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        plot_losses = []
        # scheduler_counter = 0
        model = net
        for i in range(epochs):
            print("\n"+'The teacher',i+1,'time is training')
            # training
            avg_meters = {
                'loss':AverageMeter(),
                'acc':AverageMeter(),
            }
            correct = 0
            total = 0
            running_loss = 0
            scheduler_counter = 0
            model.train()
            teacher.eval()
            process = tqdm(total=len(trainloader))
            for images, labels,_ in trainloader:
                loss = 0
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                labels_pred  = model(images)
                with torch.no_grad():
                    teach_pred  = teacher(images.to(DEVICE))
                # print(images.shape)
                # print(labels.shape)
                # print(labels_pred.shape)
                # loss = criterion(labels_pred, labels)
                loss = loss_fn(student_output=labels_pred,
                               teacher_output=teach_pred,
                               labels=labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    labels_pred = torch.argmax(labels_pred, dim=1)
                    correct_sum = (labels_pred == labels).sum().item()
                    correct += correct_sum
                    total += labels_pred.size(0)
                    running_loss += loss.item()

                avg_meters['loss'].update(loss.item(), images.size(0))
                avg_meters['acc'].update(correct_sum/(labels_pred.size(0)*256*256))
                # print(loss.item())
                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('acc', avg_meters['acc'].avg),
                ])

                process.set_postfix(postfix)
                process.update(1)
            process.close()
            # exp_lr_scheduler.step()
            exp_lr_scheduler.step()
            epoch_loss = running_loss / len(trainloader.dataset)
            epoch_acc = correct / (total*256*256)
            print('loss:',avg_meters['loss'].avg,',lossOld:',epoch_loss)
            print('acc:',avg_meters['acc'].avg,',accOld:',epoch_acc)
            losslist.append(avg_meters['loss'].avg)

    def test(net, testloader):
        """Validate the model on the test set."""
        model = net
        model.eval()
        test_correct = 0
        test_loss = 0
        test_total = 0
        test_running_loss = 0
        avg_metersT = {
            'loss':AverageMeter(),
            'acc':AverageMeter(),
        }
        with torch.no_grad():
            process = tqdm(total=len(testloader))
            for images, labels ,img_ids in testloader:
                # if (mps == True):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                test_loss = criterion(outputs,labels)

                outputs = torch.argmax(outputs, dim=1)
                # labels = labels.to(DEVICE)
                correct_sum = (outputs == labels).sum().item()
                test_correct += correct_sum
                test_total += labels.size(0)
                test_running_loss += test_loss.item()

                avg_metersT['loss'].update(test_loss.item(), images.size(0))
                avg_metersT['acc'].update(correct_sum/(labels.size(0)*256*256), images.size(0))
                postfix = OrderedDict([
                    ('loss', avg_metersT['loss'].avg),
                    ('acc', avg_metersT['acc'].avg),

                ])
                process.set_postfix(postfix)
                process.update(1)
            process.close()
        epoch_test_loss = test_running_loss / len(testloader.dataset)
        epoch_test_acc = test_correct / (test_total*256*256)
        print('test_lossT： ', epoch_test_loss,
              'test_accuracyT:',epoch_test_acc
              )
        print('test_loss： ', avg_metersT['loss'].avg,
              'test_acc:',avg_metersT['acc'].avg,
              )
        plot_examples(image=images,label=labels,pre=outputs,client=client)
        return avg_metersT['loss'].avg,avg_metersT['acc'].avg
        # criterion = torch.nn.CrossEntropyLoss()
        # correct,total, loss = 0,0, 0.0
        # with torch.no_grad():
        #     for images, labels in tqdm(testloader):
        #         outputs = net(images.to(DEVICE))
        #
        #         labels = labels.to(DEVICE)
        #         loss += criterion(outputs, labels).item()
        #         correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        # accuracy = correct / len(testloader.dataset)
        # print('accu:',accuracy,
        #       'loss:',loss)
        # return loss, accuracy
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
            train(net,teacher,trainloader,epochs)
            return self.get_parameters(config={}), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            print("\nEvaluation started on Client...")
            net.eval()
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            print(loss,accuracy)
            return loss, len(testloader.dataset), {"accuracy": accuracy}

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )
    EndTime = datetime.now()
    print(losslist)
    saveData(new_data=losslist, filename=filename,args=args)
    if args.dataset == "Chest":
        picName='CHNCXR_0001_0'
    elif args.dataset == "CVC":
        picName='1'
    onePicTrain(picName=picName,filename=filename,args=args,model=net,DEVICE=DEVICE)
    print ("StartTime is %s" % StartTime)
    print("EndTime is %s" % EndTime)
    UseTime = (EndTime - StartTime)
    print("UseTime is %s" % UseTime)
