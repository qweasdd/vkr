import math
import numpy as np
import torch
import torch.optim as optim

def coords_transform(img, line):
    x1 = math.floor(line['L_Width'] * img.shape[1])
    y1 = math.floor(line['L_Height'] * img.shape[0])
    x2 = math.ceil(line['R_Width'] * img.shape[1])
    y2 = math.ceil(line['R_Height'] * img.shape[0])
    
    return [[x1, y1, x2, y2]]

def IoU(predict, y):
    iou_sum = 0
    
    for i in range(predict.shape[0]):
        predict_line = predict[i]
        y_line = y[i]

        int_x1 = max(predict_line[0], y_line[0])
        int_y1 = max(predict_line[1], y_line[1])
        int_x2 = min(predict_line[2], y_line[2])
        int_y2 = min(predict_line[3], y_line[3])

        int_area = (int_x2 - int_x1) * (int_y2 - int_y1)

        union_area = max(predict_line[2] - predict_line[0], 0) * max(predict_line[3] - predict_line[1], 0) + (y_line[2] - y_line[0]) * (y_line[3] - y_line[1])  - int_area
        
        iou_sum += int_area * 1.0 / union_area
        
    return iou_sum / predict.shape[0]

def iou_metric(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1).astype('int')
    labels = labels.squeeze(1).astype('int')
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean()

def dice_metirc(outputs, labels):
    sum_dice = 0
    for i in range(outputs.shape[0]):
        seg = outputs[i]
        gt = labels[i]
        sum_dice += (np.sum(seg[gt == 1]) * 2.0 + 1e-6) / (np.sum(seg) + np.sum(gt) + 1e-6)

    return sum_dice / outputs.shape[0]

def iou_metric(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1).astype('int')
    labels = labels.squeeze(1).astype('int')
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean()

def dice_metirc(outputs, labels):
    sum_dice = 0
    for i in range(outputs.shape[0]):
        seg = outputs[i]
        gt = labels[i]
        sum_dice += (np.sum(seg[gt == 1]) * 2.0 + 1e-6) / (np.sum(seg) + np.sum(gt) + 1e-6)

    return sum_dice / outputs.shape[0]

def euclidean_dist(x, y):
    
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def coll(data):
    images, classes = zip(*data)
    
    return torch.cat(images, dim = 0), torch.cat(classes, dim = 0)

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

class CyclicLR(object):
   

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs