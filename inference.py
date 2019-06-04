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
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

def data_to_matrix(class_model, det_model, ver_model, data, data_path, gpu, verbose = False):
    
    result_matrix = np.zeros((data.shape[0], 512), dtype = 'float16')
    
    for n, i in enumerate(data.iterrows()):
        row = i[1]

        res_vector = row_to_vector(class_model, det_model, ver_model, row, data_path, gpu)
        result_matrix[n] = res_vector
        
        if verbose and n % 100 == 0:
            print(n, '/', data.shape[0])
        
    return result_matrix   

def row_to_vector(class_model, det_model, ver_model, row, data_path, gpu):
    imgs = ast.literal_eval(row['downloaded_images'])
    
    resized_images, original_images = load_images(imgs, data_path)
    
    class_predictions = classification(class_model, resized_images, gpu)
    
    if len(class_predictions) == 0:
        return torch.zeros((512), dtype = torch.float16)
        
    detection_predictions = detection(det_model, resized_images, class_predictions, gpu)
    
    croped_imgs_tensor = []
    for n, bounds in enumerate(detection_predictions):
        orig = original_images[class_predictions[n]]

        original_height, original_width, _ = orig.shape


        bounds[0] = max(0, bounds[0] - 0.02)
        bounds[1] = max(0, bounds[1] - 0.02)
        bounds[2] = min(1, bounds[2] + 0.02)
        bounds[3] = min(1, bounds[3] + 0.02)



        x = bounds[0] * original_width
        y = bounds[1] * original_height
        width = (bounds[2] - bounds[0]) * original_width
        height = (bounds[3] - bounds[1]) * original_height

        crop = orig[int(y):int(y) + int(height), int(x): int(x + width)]
        crop = cv2.resize(crop, (224, 224))


        croped_imgs_tensor.append(transform(crop))
        
    ver_input = torch.stack(croped_imgs_tensor)
    if gpu:
        ver_input = ver_input.to('cuda')
        
    result_vector = ver_model(ver_input).cpu().detach().numpy()
    
    return result_vector

def classification(class_model, images, gpu):
    
    torch_images = []
    for img in images:
        torch_images.append(transform(img))
        
    class_input = torch.stack(torch_images)
    if gpu:
        class_input = class_input.to('cuda')
        
    output = torch.sigmoid(class_model(class_input)).cpu().detach().numpy()
    good_indecies = (output >= 0.6).nonzero()[0]
    
    return good_indecies

def detection(det_model, images, good_indecies, gpu):
    
    torch_images = []
    for i in good_indecies:
        torch_images.append(transform(images[i]))

    det_input = torch.stack(torch_images)
    
    if gpu:
        det_input = det_input.to('cuda')
    
    det_output = torch.sigmoid(det_model(det_input)).cpu().detach().numpy()
    
    return det_output

def load_images(imgs, data_path):
    resized_imgs = []
    original_imgs = []
    
    for name in imgs:
        
        original_image = plt.imread(data_path + name)
        
        if len(original_image.shape) < 3:
            original_image = np.repeat(np.expand_dims(original_image, axis = 2), axis = 2, repeats = 3)
        if original_image.shape[2] == 4:
            original_image = np.repeat(original_image[:, :, 0:1], axis = 2, repeats = 3)
        
        
        original_imgs.append(original_image)
        
        temp_img = cv2.resize(original_image, (224, 224))
        if len(temp_img.shape) < 3:
            temp_img = np.repeat(np.expand_dims(temp_img, axis = 2), axis = 2, repeats = 3)
        if temp_img.shape[2] == 4:
            temp_img = np.repeat(temp_img[:, :, 0:1], axis = 2, repeats = 3)
        resized_imgs.append(temp_img)
        
    return resized_imgs, original_imgs