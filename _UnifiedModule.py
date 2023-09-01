import argparse
import torchvision.transforms as transforms
from DataSet import Dataset
from torch.utils import data
import torch
import glob
from glob import glob
import os
import cv2
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

#tifPicShow
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#unet++
import albumentations as A
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

# __all__ = [ 'BCEDiceLoss']

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, default = "",required=True, help="path to your train dataset")
    # #Train
    # parser.add_argument("--test", type=str, default = "",help="path to your test dataset")
    #Test
    parser.add_argument("--client", type=str, default="train", help="path to your data")
    parser.add_argument("--cuda", type=str, default="cuda2", help="path to your data")
    parser.add_argument("--model", type=str, default="simpleUnet", help=" simpleUnet | resUnet | transUnet ")
    parser.add_argument("--dataset", type=str, default="Chest", help="name to be appended to checkpoints")
    parser.add_argument("--picFormat",type=str, default=".png",help="name your datas' format")
    # parser.add_argument("--loss",type=str,default="crossentropy",
    #                     help="focalloss | iouloss | crossentropy",)#Loss
    parser.add_argument("--num_epochs", type=int, default=20, help="dnumber of epochs")
    parser.add_argument('--num_classes', default=1, type=int,help='number of classes')
    parser.add_argument('--input', default=256, type=int,help='image width')
    parser.add_argument('--vit_name', type=str,default='R50-ViT-B_16', help='select one vit model')
    # #Epochs
    # parser.add_argument("--batch", type=int, default=1, help="batch size")
    # #batch_size
    # parser.add_argument("--save_step", type=int, default=5, help="epochs to skip")
    return parser.parse_args()

def load_data(args):
    client = args.client
    dataset = args.dataset
    picFormatName =args.picFormat
    t = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor(),])
    # imgs_path = glob.glob(r'../../Experiment04_Unet in Medical segmentation/raw/*')
    # imgs_path.sort()
    # labels_path = glob.glob(r'../../Experiment04_Unet in Medical segmentation/labels/*')
    # labels_path.sort()

    # imgs_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/images/*')
    # imgs_path.sort()
    # labels_path = glob.glob(r'Data/Chest_Xray_Masks_and_Labels/masks/0/*')
    # labels_path.sort()
    if dataset == 'Chest':
        dataPath = 'Chest_Xray_Masks_and_Labels'
    elif dataset == 'CVC':
        dataPath = 'CVC_ClinicDB'
    else:
        print('Dataset is Error!Not Choice')
    if picFormatName == '.png':
        picFormat = '.png'
    elif picFormatName == '.tif':
        picFormat = '.tif'
    else:
        print('Datasets format is Error!Not Choice')

    img_ids = glob(os.path.join('Data', dataPath, 'images', '*' + picFormat))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    length = int(len(img_ids))
    print(length)
    val_img_ids   = img_ids[int(length*0.8):]

    img_ids = img_ids[:int(length*0.8)]
    print(len(img_ids))
    train_id1,train_id2_3 = train_test_split(img_ids,test_size=0.66,random_state = 42)
    train_id2,train_id3   = train_test_split(train_id2_3,test_size=0.5,random_state = 42)
    if client == "train":
        train_img_ids = img_ids
    elif client == "client1":
        # train_img_ids, _ = train_test_split(img_ids, test_size=0.8,random_state=16)
        train_img_ids = train_id1
    elif client == "client2":
        train_img_ids = train_id2
    elif client == "client3":
        train_img_ids = train_id3
    print(len(train_img_ids))
    print(len(val_img_ids))

    BACH_SIZE=4
    train_transform = t
    val_transform   = t
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('Data', dataPath, 'images'),
        mask_dir=os.path.join('Data', dataPath, 'masks'),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= picFormat,
        mask_ext= picFormat,
        # num_classes=config['num_classes'],
        num_classes= 1 ,
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('Data', dataPath, 'images'),
        mask_dir=os.path.join('Data', dataPath, 'masks'),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= picFormat,
        mask_ext= picFormat,
        # num_classes=config['num_classes'],
        num_classes= 1 ,
        transform=val_transform)
    trainset = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=True,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=True)
    print(len(trainset))
    testset = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size= BACH_SIZE,
        shuffle=False,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=False)
    print(len(testset))
    # TrainDataSet = segDataset(img_paths = train_path,anno_paths = train_labels_path,transform = t)
    # TestDataSet  = segDataset(img_paths = test_path,anno_paths = test_labels_path,transform = t)
    # trainset = data.DataLoader(TrainDataSet,batch_size=BACH_SIZE,
    #                            shuffle=True, num_workers=0)#num_works =8
    # testset = data.DataLoader( TestDataSet, batch_size=BACH_SIZE,
    #                            shuffle=False, num_workers=0)#num_works =8

    print('-----------------Load Data---------------------------')
    return trainset, testset

def onePicTrain(picName,filename,args,model,DEVICE):
    dataset = args.dataset
    picFormat = args.picFormat
    client = args.client
    type = filename

    #imagePic
    if dataset == 'Chest':
        dataPath = 'Chest_Xray_Masks_and_Labels'
        img_id = 'Data/'+ dataPath+'/images/'+ picName+ picFormat
        img = mpimg.imread(img_id)
        plt.imshow(img)
    elif dataset == 'CVC':
        dataPath = 'CVC_ClinicDB'
        img_id = 'Data/'+ dataPath+'/images/'+ picName+ picFormat
        img = tiff.imread(img_id)
        plt.imshow(show_tif(img[:,:,:3]))
    else:
        print('The Dataset Name is Error!')
    plt.savefig(os.path.join('PicList',type,'image.jpg'))

    #train
    test_img_ids = []
    test_img_ids.append(picName)
    test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),])
    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('Data', dataPath, 'images'),
        mask_dir=os.path.join('Data', dataPath, 'masks'),
        # img_ext=config['img_ext'],
        # mask_ext=config['mask_ext'],
        img_ext= picFormat,mask_ext= picFormat,
        # num_classes=config['num_classes'],
        num_classes= 1 ,transform=test_transform)
    tloader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=config['batch_size'],
        batch_size= 1,#!!!!!!!!!
        shuffle=False,
        # num_workers=config['num_workers'],
        num_workers= 0,
        drop_last=False)
    criterion = nn.CrossEntropyLoss()#loss_fn
    model.eval()
    with torch.no_grad():
        for images,labels,img_ids in tloader:
            # if (mps == True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_loss = criterion(outputs,labels)
            print('test_loss： ', test_loss)

    #labelPic
    original_tensor = labels[0].float()
    unloader = transforms.ToPILImage()
    pic = original_tensor.cpu().clone()  # clone the tensor
    pic = pic.squeeze(0)  # remove the fake batch dimension
    pic = unloader(pic)
    pic.save(os.path.join('PicList',type,'label.png'))

    #prePic
    original_tensor = outputs[0].float()
    anno_tensor = original_tensor
    if dataset == 'Chest':
        print('is the Chest')
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)
    else:
        print('is the CVC')
    anno_tensor[anno_tensor > 0] = 1
    unloader = transforms.ToPILImage()
    pic = anno_tensor.to(DEVICE).clone()
    pic = pic.squeeze(0).float()
    # print(pic)
    pic = unloader(pic)
    pic.save(os.path.join('PicList',type,client+'.png'))
    # if client == "client1":
    #     pic.save('PicList/pic/pre1.jpg')
    # elif client == "client2":
    #     pic.save('PicList/pic/pre2.jpg')
    # elif client == "client3":
    #     pic.save('PicList/pic/pre3.jpg')
    # else:
    #     pic.save('pre.jpg')

def saveData(new_data,filename,args):
    client = args.client
    model = args.model
    type = filename
    dataPath = os.path.join('DataList',type,client+'.txt')
    print(dataPath)
    # if client == "client1":
    #     dataPath = 'DataList/Unet/dataTest1.txt'
    # elif client == "client2":
    #     dataPath = 'DataList/Unet/dataTest2.txt'
    # elif client == "client3":
    #     dataPath = 'DataList/Unet/dataTest3.txt'
    with open(dataPath, 'a') as file:
        # 写入新数据
        for num in new_data:
            file.write(str(num) + ',')
        file.write('----')
            
def show_tif(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




#loss
def dice_loss(input, target):
    smooth = 1e-5
    input = torch.sigmoid(input)
    num = target.size(0)
    input = input.view(num, -1)
    target = target.view(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return dice_loss(input, target)


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        return bce




class LogCoshBDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        bce = F.binary_cross_entropy_with_logits(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        x = bce + dice
        return x

#acc

# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#
#     return (intersection + smooth) / (union + smooth)
def iou_score(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    # intersection = (output * target).sum()
    intersection = output.dot(target)

    return (intersection + smooth) / \
        (output.sum() + target.sum() - intersection + smooth)
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def ppv_ppv(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / (output.sum() + smooth)