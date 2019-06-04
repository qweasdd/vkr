from datasets import DetectionDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import time
from utils import coords_transform, IoU
import argparse


parser = argparse.ArgumentParser(description='inf')
parser.add_argument('--train-path', type=str, default='data/det_train.csv')
parser.add_argument('--test-path', type=str, default='data/det_test.csv')
parser.add_argument('--data-path', type=str, default='data/det_images/')
parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true')
parser.add_argument('--batch-size', type=int, default=32)
parser.set_defaults(use_cuda = False)
args = parser.parse_args()



gpu = args.use_cuda

train_data = DetectionDataset(args.train_path,args.data_path, train = True)
test_data = DetectionDataset(args.test_path, args.data_path ,train = False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers= 6)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16)

model = torchvision.models.resnet18(pretrained= True)
model.fc = nn.Linear(512, 4)
if gpu:
    model = model.to('cuda')
    
optimizer = optim.Adam(model.parameters(), lr= 1e-3)
mse_loss = nn.MSELoss()

epochs = 35
steps_per_epoch = len(train_data) / args.batch_size
steps = int(steps_per_epoch * epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = steps, eta_min= 1e-5)

print('Training:\n')

for i in range(epochs):
    
    model.train()
    sum_loss = 0
    iou = 0
    
    
    for n, (x, y) in enumerate(train_loader):
        if gpu:
            x, y = x.to('cuda'), y.to('cuda')
        
        output = torch.sigmoid(model(x))
        
        optimizer.zero_grad()
        
        loss =  mse_loss(output, y)
        
        
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        sum_loss += loss.item()
        
        predict = output.cpu().detach().numpy().clip(min = 0, max = 1)
        y = y.cpu().detach().numpy()
        iou += IoU(predict, y)
    
    print(time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
    print('Epoch : %d Train loss : %.5f Train metric : %.4f'%(i, sum_loss / (n + 1), iou / (n + 1)))
    
    model.eval()
    sum_loss = 0
    iou = 0
          
    with torch.no_grad():
        for n, (x, y) in enumerate(test_loader):
            
            if gpu:
                x, y = x.to('cuda'), y.to('cuda')

            output = torch.sigmoid(model(x))
            
        
            loss = mse_loss(output, y)
            sum_loss += loss.item()

            predict = output.cpu().detach().numpy().clip(min = 0, max = 1)
            y = y.cpu().detach().numpy()
            iou += IoU(predict, y)
        
        print('Epoch : %d Train loss : %.5f Train metric : %.4f'%(i, sum_loss / (n + 1), iou / (n + 1)))
        print()
        
torch.save(model.state_dict(), 'detection_model')