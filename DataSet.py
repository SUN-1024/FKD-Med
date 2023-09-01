import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import ndimage
import glob

# from utils.image_utils import get_image_names, get_image_and_mask
import matplotlib.pyplot as plt
import PIL.Image as Image
import os


# imgs_path = glob.glob(r'../../Experiment04_Unet in Medical segmentation/raw/*')
# imgs_path.sort()
# labels_path = glob.glob(r'../../Experiment04_Unet in Medical segmentation/labels/*')
# labels_path.sort()
#
# i = int(len(imgs_path)*0.8)
# train_path = imgs_path[ :i]
# train_labels_path = labels_path[ :i]
# test_path = imgs_path[i: ]
# test_labels_path = labels_path[i: ]

# imgs_path = glob.glob(r'Data/ISIC2018_Task1-2_Training_Input/*')
# imgs_path.sort()
# labels_path = glob.glob(r'Data/ISIC2018_Task1_Training_GroundTruth/*')
# labels_path.sort()
# i = int(len(imgs_path)*0.8)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        """相关参数定义创建"""
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes#toMask
        self.transform = transform
    """获取文件夹中图片的数量即获取数据集的长度"""
    def __len__(self):
        return len(self.img_ids)
    """从文件夹中读取图片，并做一些需要的图片处理操作"""
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        if self.img_ext == '.png':
            img = Image.open(img_path)
        elif self.img_ext == '.jpg':
            img = Image.open(img_path)
        elif self.img_ext == '.tif':
            img = Image.fromarray(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
        else:
            print("Couldn't read the Pic Format")
            
        img = img.convert('L')
        img_tensor = self.transform(img)

        mask = os.path.join(self.mask_dir, '0',img_id + self.mask_ext)
        mask = Image.open(mask)
        mask_tensor = self.transform(mask)
        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)
        mask_tensor[mask_tensor > 0] = 1
        # mask = []
        # for i in range(self.num_classes):
        #     mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
        #                                         img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        #
        # mask = np.dstack(mask)
        # if self.transform is not None:
        #     augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
        #     img = augmented['image']#参考https://github.com/albumentations-team/albumentations
        #     mask = augmented['mask']
        # print('img:',img)
        # print('mask:',mask)
        # img = img.astype('float32') / 255
        # img = img.transpose(2, 0, 1)
        # mask = mask.astype('float32') / 255
        # mask = mask.transpose(2, 0, 1)

        return img_tensor, mask_tensor, {'img_id': img_id}

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
#         """
#         Args:
#             img_ids (list): Image ids.
#             img_dir: Image file directory.
#             mask_dir: Mask file directory.
#             img_ext (str): Image file extension.
#             mask_ext (str): Mask file extension.
#             num_classes (int): Number of classes.
#             transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
#
#         Note:
#             Make sure to put the files as the following structure:
#             <dataset name>
#             ├── images
#             |   ├── 0a7e06.jpg
#             │   ├── 0aab0a.jpg
#             │   ├── 0b1761.jpg
#             │   ├── ...
#             |
#             └── masks
#                 ├── 0
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 |
#                 ├── 1
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 ...
#         """
#         """相关参数定义创建"""
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#         self.num_classes = num_classes
#         self.transform = transform
#     """获取文件夹中图片的数量即获取数据集的长度"""
#     def __len__(self):
#         return len(self.img_ids)
#     """从文件夹中读取图片，并做一些需要的图片处理操作"""
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#
#         img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
#         print(os.path.join(self.img_dir, img_id + self.img_ext))
#         mask = []
#         for i in range(self.num_classes):
#             mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
#                                                 img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
#
#         mask = np.dstack(mask)
#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
#             img = augmented['image']#参考https://github.com/albumentations-team/albumentations
#             mask = augmented['mask']
#         # print('img:',img)
#         # print('mask:',mask)
#         img = img.astype('float32') / 255
#         img = img.transpose(2, 0, 1)
#         mask = mask.astype('float32') / 255
#         mask = mask.transpose(2, 0, 1)
#
#         return img, mask, {'img_id': img_id}