from datasets import SegmentationDataset
from losses import SoftDiceLoss, lovasz_hinge
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import time
from utils import coords_transform, IoU, iou_metric, dice_metirc
import argparse
from models import SegModel

        


parser = argparse.ArgumentParser(description='inf')
parser.add_argument('--train-path', type=str, default='data/segm_train.csv')
parser.add_argument('--test-path', type=str, default='data/segm_test.csv')
parser.add_argument('--images-path', type=str, default='data/segm_images/')
parser.add_argument('--masks-path', type=str, default='data/segm_masks/')
parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true')
parser.add_argument('--batch-size', type=int, default=16)
parser.set_defaults(use_cuda = False)
args = parser.parse_args()


gpu = args.use_cuda

train_data = SegmentationDataset(args.train_path, args.images_path, args.masks_path,train = True)
test_data = SegmentationDataset(args.test_path, args.images_path, args.masks_path,train = True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers= 6)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16)

model = SegModel()

if gpu:
    model = model.to('cuda')
    
optimizer = optim.Adam(model.parameters(), lr= 1e-4)
bce_loss = nn.BCEWithLogitsLoss()
soft_dice_loss = SoftDiceLoss()

epochs = 30
steps_per_epoch = len(train_data) / args.batch_size
steps = int(steps_per_epoch * epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = steps, eta_min= 1e-6)

print('Training:\n')

for i in range(epochs):
    
    model.train()
    sum_loss = 0
    iou_sum = 0
    dice_sum = 0
    
    for n, (x, y) in enumerate(train_loader):
        
        mask = y.numpy()
        if gpu:
            x, y = x.to('cuda'), y.to('cuda')
        
        output = model(x)
        
        optimizer.zero_grad()
        
        if i < 20:
            loss =  bce_loss(output, y) - torch.log(soft_dice_loss(output, y))
        else:
            loss =  lovasz_hinge(output.squeeze(), y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        sum_loss += loss.item()
        
        predict = torch.sigmoid(output).cpu().detach().numpy()
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        
        iou_sum += iou_metric(predict, mask)
        dice_sum += dice_metirc(predict, mask)
        
    print(time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
    print('Epoch : %d Train loss : %.5f Train metric : %.4f %.4f'%(i, sum_loss / (n + 1), iou_sum / (n + 1), dice_sum / (n + 1)))
    
    model.eval()
    sum_loss = 0
    iou_sum = 0
    dice_sum = 0
          
    with torch.no_grad():
        for n, (x, y) in enumerate(test_loader):
            
            mask = y.numpy()
            if gpu:
                x, y = x.to('cuda'), y.to('cuda')

            output = model(x)
        
            if i < 20:
                loss =  bce_loss(output, y) - torch.log(soft_dice_loss(output, y))
            else:
                loss =  lovasz_hinge(output.squeeze(), y.squeeze())
            
            sum_loss += loss.item()

            predict = torch.sigmoid(output).cpu().detach().numpy()
            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0

            iou_sum += iou_metric(predict, mask)
            dice_sum += dice_metirc(predict, mask)
        
        print('Epoch : %d Test loss : %.5f Test metric : %.4f  %.4f'%(i, sum_loss / (n + 1), iou_sum / (n + 1), dice_sum / (n + 1)))
        print()
        
torch.save(model.state_dict(), 'segmentation_model')