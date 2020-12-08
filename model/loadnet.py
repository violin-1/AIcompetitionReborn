import os
from model import alex,captcha,resnet,pspnet
import dataloader
import torch

dataset = dataloader.CaptchaDataSet.CaptchaData(premode=0)

def ResNet(model_path = '.',loadbest = False):
    net = resnet.ResNet(resnet.ResidualBlock, [2, 2, 2], num_classes=len(dataset.captcha_list), model_path=model_path)
    if(loadbest):
        path = os.path.join(model_path,'model','BestResNet.pth')
    else:
        path = os.path.join(model_path,'model','ResNet.pth')
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    return net

def Captcha(model_path = '.',loadbest = False):
    net = captcha.Net(model_path,len(dataset.captcha_list))
    if(loadbest):
        path = os.path.join(model_path,'model','BestCaptcha.pth')
    else:
        path = os.path.join(model_path,'model','Captcha.pth')
    # path = os.path.join(model_path,'model','Captcha.pth')
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    return net

def PspNet(model_path = '.',loadbest = False):
    net = pspnet.PspNet(pspnet.PspNetBlock, [2, 2, 2], num_classes=len(dataset.captcha_list), model_path=model_path)
    if(loadbest):
        path = os.path.join(model_path,'model','BestPspNet.pth')
    else:
        path = os.path.join(model_path,'model','PspNet.pth')
    # path = os.path.join(model_path,'model','Captcha.pth')
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    return net