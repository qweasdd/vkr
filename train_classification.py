from datasets import ClassificationDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
import numpy as np
import time
import argparse



parser = argparse.ArgumentParser(description='inf')
parser.add_argument('--train-path', type=str, default='data/class_train.csv')
parser.add_argument('--test-path', type=str, default='data/class_test.csv')
parser.add_argument('--data-path', type=str, default='data/class_images/')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true')
parser.set_defaults(use_cuda = False)
args = parser.parse_args()

gpu = args.use_cuda

train_data = ClassificationDataset(args.train_path,args.data_path, train = True)
test_data = ClassificationDataset(args.test_path, args.data_path ,train = False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers= 6)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 16)

model = torchvision.models.resnet18(pretrained= True)
model.fc = nn.Linear(512, 1)
if gpu:
    model = model.to('cuda')
    
optimizer = optim.Adam(model.parameters(), lr= 1e-3)
bce_loss = nn.BCEWithLogitsLoss()

epochs = 30

steps_per_epoch = len(train_data) / args.batch_size
steps = int(steps_per_epoch * epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = steps, eta_min= 1e-5)

print('Training:\n')


for i in range(epochs):
    
    model.train()
    sum_loss = 0
    correct = 0
    f1 = 0
    
    predict_list = []
    y_list = []
    
    for n, (x, y) in enumerate(train_loader):
        if gpu:
            x, y = x.to('cuda'), y.to('cuda')
        
        logits = model(x)
        
        output = torch.sigmoid(logits)
        predict = output.cpu().detach().numpy()
        
        optimizer.zero_grad()
        
        loss =  bce_loss(logits, y)
        
        
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        sum_loss += loss.item()
        
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        
        y = y.cpu().detach().numpy()
        correct += (predict == y).sum()
        
        predict_list.append(predict)
        y_list.append(y)
        
    predict_list = np.concatenate(predict_list, axis = 0)    
    y_list = np.concatenate(y_list, axis = 0)      
    f1 = f1_score(y_list, predict_list)
    
    print(time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
    print('Epoch : %d Train loss : %.5f Train metric : %.4f  %.4f'%(i, sum_loss / (n + 1), correct * 1.0 / len(train_data), f1))
    
    model.eval()
    sum_loss = 0
    correct = 0
    f1 = 0
    predict_list = []
    y_list = []
    
    with torch.no_grad():
        for n, (x, y) in enumerate(test_loader):
            
            if gpu:
                x, y = x.to('cuda'), y.to('cuda')

            logits = model(x)
        
            output = torch.sigmoid(logits)
            predict = output.cpu().detach().numpy()
        
            loss =  bce_loss(logits, y)
            sum_loss += loss.item()

            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0

            y = y.cpu().detach().numpy()
            correct += (predict == y).sum()
            
            predict_list.append(predict)
            y_list.append(y)
            
        predict_list = np.concatenate(predict_list, axis = 0)    
        y_list = np.concatenate(y_list, axis = 0)      
        f1 = f1_score(y_list, predict_list)
        
        print('Epoch : %d Test loss : %.5f Test metric : %.4f  %.4f'%(i, sum_loss / (n + 1), correct * 1.0 / len(test_data), f1 ))
        print()
        
torch.save(model.state_dict(), 'classification_model')