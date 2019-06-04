import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import cosine_sim



class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class SegModel(nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        
        is_deconv = False
        
        resnet = torchvision.models.resnet18(pretrained= True)
        
        self.conv0 = resnet.conv1
        self.bn0 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.pool = nn.MaxPool2d(2, 2, ceil_mode= True)
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4
        
        self.dec5 = DecoderBlockV2(512, 512, 256, is_deconv)
        self.dec4 = DecoderBlockV2(256 + 256, 512, 256, is_deconv)
        self.dec3 = DecoderBlockV2(128 + 256, 256, 64, is_deconv)
        self.dec2 = DecoderBlockV2(64 + 64, 128, 128, is_deconv)
        self.dec1 = DecoderBlockV2(128 + 64, 128, 32, is_deconv)
        self.dec0 = ConvRelu(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        out1 = F.relu(x)
        x = self.maxpool(out1)
        
        out2 = self.layer1(x)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        
        

        dec5 = self.dec5(out5)

        dec4 = self.dec4(torch.cat([dec5, out4], 1))
        dec3 = self.dec3(torch.cat([dec4, out3], 1))
        dec2 = self.dec2(torch.cat([dec3, out2], 1))
        dec1 = self.dec1(torch.cat([dec2, out1], 1))
        
        
        
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return x_out
    
    
class MarginCosineProduct(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)

        return output

class VerificationModel(nn.Module):
    def __init__(self, num_class):
        super(VerificationModel, self).__init__()
        resnet = torchvision.models.resnet18(pretrained= True)
        self.pas = nn.Sequential(*list(resnet.children())[:-1])
        
        
        self.bn = nn.BatchNorm1d(512)
        self.bn.bias.requires_grad_(False)
        
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0.)
        
        self.margin = MarginCosineProduct(512, num_class)
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def forward(self, x, label):
        

        x = torch.cat([
            (x[:, [0]] - self.mean[0]) / self.std[0],
            (x[:, [1]] - self.mean[1]) / self.std[1],
            (x[:, [2]] - self.mean[2]) / self.std[2],
        ], 1)
        
        dense = self.pas(x).squeeze(2).squeeze(2)
        features = self.bn(dense)
        
        logits = self.margin(features, label)
        
        return dense, logits
    
    
class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        resnet = torchvision.models.resnet18(pretrained= True)
        self.pas = nn.Sequential(*list(resnet.children())[:-1])
        
        
        
        self.bn = nn.BatchNorm1d(512)
        self.bn.bias.requires_grad_(False)
        
        
        
    def forward(self, x):
        
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
            (x[:, [0]] - mean[0]) / std[0],
            (x[:, [1]] - mean[1]) / std[1],
            (x[:, [2]] - mean[2]) / std[2],
        ], 1)
        
        dense = self.bn(self.pas(x).squeeze(2).squeeze(2).mean(dim = 0, keepdim = True)).squeeze(0)
        dense = dense / torch.norm(dense, 2)
        
        return dense