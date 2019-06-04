import torch
import torchvision
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
import albumentations as albu
import matplotlib.pyplot as plt
import ast
import numpy as np
from utils import coords_transform

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_path, train = True):
        if train:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = pd.read_csv(csv_path)
        
        self.transforms = transforms.ToTensor()
        
        self.data_path = data_path
        
        self.flip = albu.HorizontalFlip()
        self.train = train
        rotate_crop = albu.Compose([albu.Rotate(limit = 10, p = 1.), albu.RandomSizedCrop((185, 202), height= 224, width= 224, p = 1.)], p = 0.95)
        self.aug = albu.Compose([rotate_crop, albu.RandomBrightnessContrast(), albu.Blur(), albu.GaussNoise()], p = 1.)
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img = plt.imread(self.data_path + self.data.iloc[idx]['Image'])
        
        if self.train:
            img = self.flip(image = img)['image']
            img = self.aug(image = img)['image']
    
        img = self.transforms(img)
        
        return img, torch.FloatTensor([self.data.iloc[idx]['Car']])
    
class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_path, train = True):
        
        
        if train:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = pd.read_csv(csv_path)
        
        self.transforms = transforms.ToTensor()
        
        self.data_path = data_path
        
        self.train = train
        rotate_crop = albu.Compose([albu.Rotate(limit = 10, p = 1.), albu.RandomSizedCrop((185, 202), height= 224, width= 224, p = 1.)], p = 0.5)
        color = albu.OneOf([albu.RandomBrightnessContrast(), albu.Blur(blur_limit = 3), albu.GaussNoise()], p = 0.3)
        self.aug = albu.Compose([albu.HorizontalFlip(), albu.RandomSizedBBoxSafeCrop(height= 224, width= 224, erosion_rate= 0.6, p = 0.45), color], bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']},  p = 1.)
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img = plt.imread(self.data_path + self.data.iloc[idx]['Image'])
        q = self.data.iloc[idx]
        bbox = coords_transform(img, q)
        
        if self.train:
            flag = True
            while flag:
                res_aug = self.aug(image = img, bboxes = bbox, category_id= [0])
                img = res_aug['image']
                new_bbox = res_aug['bboxes']
                
                if len(new_bbox) > 0:
                    flag = False
                    bbox = new_bbox
    
        img = self.transforms(img)
        bbox = torch.FloatTensor(bbox[0]) / 224
        
        return img, bbox
    
    
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_path, masks_path, train = True):
        
        if train:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = pd.read_csv(csv_path)
        
        self.transforms = transforms.ToTensor()
            
        self.images_path = images_path
        self.masks_path = masks_path
        
        self.train = train
        rotate_crop = albu.Compose([albu.Rotate(limit = 10, p = 1.), albu.RandomSizedCrop((165, 202), height= 224, width= 224, p = 1.)], p = 0.6)
        color = albu.OneOf([albu.RandomBrightnessContrast(), albu.Blur(blur_limit = 3), albu.GaussNoise()], p = 0.6)
        self.aug = albu.Compose([albu.HorizontalFlip(), albu.RandomSizedBBoxSafeCrop(height= 224, width= 224, erosion_rate= 0.45, p = 0.3), color], bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']},  p = 1.)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        name = self.data.iloc[idx]['Image']
        img = plt.imread(self.images_path + name)
        mask = plt.imread(self.masks_path + name[:name.rfind('.')] + '.png')[:, :, 0:1]
        
        min_i = img.shape[0] + 100
        min_j = img.shape[1] + 100
        max_i = -1
        max_j = -1
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i,j , 0] == 1:
                    if min_i > i:
                        min_i = i
                    if max_i < i:
                        max_i = i
                    if min_j > j:
                        min_j = j
                    if max_j < j:
                        max_j = j
                    
        bbox = [[min_j, min_i, max_j, max_i]]
        
        
        
        if self.train:
            res_aug = self.aug(image = img, mask = mask, bboxes = bbox, category_id = [0])
            img = res_aug['image']
            mask = res_aug['mask']
    
        img = self.transforms(img)
        mask = self.transforms(mask)
        
        return img, mask
    
class VerificationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_path):
        self.data = pd.read_csv(csv_path)
        self.groups = {x:y.tolist() for x, y in self.data.groupby('sdocid').groups.items()}
        self.ids = list(self.groups.keys())
        self.ad_imgs_list = dict(zip(self.data.index.tolist(), self.data.cars_images.map(lambda x : ast.literal_eval(x)).tolist()))
        
        self.path = data_path
        
        self.transform = transforms.ToTensor()
        self.class_dict = dict(zip(self.data['sdocid'].unique().tolist(), range(0, self.data['sdocid'].nunique())))
        
        
        
        self.aug = albu.Compose([albu.HorizontalFlip(),albu.RandomSizedCrop(min_max_height = (200, 215), height = 224, width = 224, p = 0.2), albu.Cutout(num_holes = 4, max_h_size= 20, max_w_size= 20,  p = 0.3)], p = 1.)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        ad_group = self.groups[self.ids[idx]]
        if len(ad_group) >= 3:
            ad_choices = np.random.choice(ad_group, size = 3, replace = False)
        else:
            ad_choices = np.random.choice(ad_group, size = len(ad_group), replace = False).tolist()
            ad_choices.extend(np.random.choice(ad_group, size = 3 - len(ad_group), replace = True))
            
        img_names = []
        for choice in ad_choices:
            img_choice = np.random.choice(self.ad_imgs_list[choice])
            img_names.append(img_choice)
            
        img0 = plt.imread(self.path + img_names[0])
        img1 = plt.imread(self.path + img_names[1])
        img2 = plt.imread(self.path + img_names[2])
        
        img0 = self.aug(image = img0)['image']
        img1 = self.aug(image = img1)['image']
        img2 = self.aug(image = img2)['image']
        
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        
        img0 = img0.unsqueeze(0)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
        img = torch.cat([img0, img1, img2], dim = 0)
        
        class_number = self.class_dict[self.ids[idx]]
        
        
        return img, torch.LongTensor([class_number] * 3)