import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import ast
import math
import numpy as np
from models import InferenceModel
import inference
import argparse

parser = argparse.ArgumentParser(description='inf')
parser.add_argument('--data-path', type=str, default='data/verification_images/')
parser.add_argument('--csv-path', type=str, default='data/test_reid.csv')
parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true')
parser.add_argument('--threshold', type=float, default=0.75)
parser.set_defaults(use_cuda = False)
args = parser.parse_args()

gpu = args.use_cuda

class_model = torchvision.models.resnet18(pretrained= True)
class_model.fc = nn.Linear(512, 1, bias = False)
class_model.load_state_dict(torch.load('models/classification_model'))
class_model = class_model.eval()

det_model = torchvision.models.resnet18(pretrained= True)
det_model.fc = nn.Linear(512, 4)
det_model.load_state_dict(torch.load('models/detection_model'))
det_model = det_model.eval()

ver_model = InferenceModel()
ver_model.load_state_dict(torch.load('models/verification_model'))
ver_model = ver_model.eval()

if gpu:
    class_model = class_model.to('cuda')
    det_model = det_model.to('cuda')
    ver_model = ver_model.to('cuda')
    
data = pd.read_csv(args.csv_path)
data_path = args.data_path

ids = data['sdocid'].tolist()
labels_pairs = []
for i in range(len(ids)):
    for j in range(len(ids)):
        if ids[i] == ids[j]:
            labels_pairs.append([i, j])

print('Working:\n')
matrix = inference.data_to_matrix(class_model, det_model, ver_model, data, data_path, gpu, True).astype('float32')

dists = np.dot(matrix, matrix.T)

threshold = args.threshold

temp = dists.copy()
    
    
temp = np.where(temp >= threshold, 1, 0)    
    
predictions_pairs = np.argwhere(temp > 0).tolist()
inter = [x for x in labels_pairs if x in predictions_pairs]
false_neg = [x for x in labels_pairs if x not in predictions_pairs]
false_acc = [x for x in predictions_pairs if x not in labels_pairs]
    
TP = len(inter)
FP = len(false_acc)
FN = len(false_neg)
TN = temp.shape[0] * temp.shape[0] - TP - FP - FN

TP -= temp.shape[0]
    
recall = (TP) / (TP + FN)
precision = (TP)/ (TP + FP)
f1 = 2 *(recall * precision) / (recall + precision)
print()
print('For threshold', threshold, ':')
print('Precision:',precision)
print('Recall:', recall)
print('F1 score:', f1)
print()
t_far = FP / (TN + FP)
print('Val:', recall)
print('False acc rate(corr):', t_far)