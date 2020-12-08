import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
# 加载所有图片，并将验证码向量化
from torch.utils.data import Dataset
from PIL import Image
import os
import csv
import torchvision.transforms as transforms

class CaptchaData(Dataset):
    def __init__(self, data_path=None, transform=transforms.Compose([transforms.ToTensor(),]), premode = 1):
        super(Dataset, self).__init__()
        self.captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.captcha_length = 4
        self.img = []
        self.img_label = {}
        if(premode == 1):
            self.transform = transform
            self.data_path = data_path
            self.img,self.img_label = self.make_dataset()
    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        # img_path, target = self.samples[index]
        img_name = self.img[index]
        img_path = os.path.join(self.data_path, img_name)
        if(self.img_label.__contains__(img_name)):
            target = self.img_label[img_name]
            target = self.text2vec(target)
            target = target.view(1, -1)[0]
        else:
            target = img_name

        img = Image.open(img_path)
        img = img.resize((160,40))
        img = img.convert('RGB') # img转成向量
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def get_img(self, index):
        out = CaptchaData(premode=0)
        out.data_path = self.data_path
        out.transform = self.transform
        for ID in index:
            out.img.append(self.img[ID])
            out.img_label[self.img[ID]] = self.img_label[self.img[ID]]

        return out
        # 验证码文本转为向量
    def text2vec(self,text):
        vector = torch.zeros((self.captcha_length, len(self.captcha_list)))

        if(text == '-1'):
            return vector

        text_len = len(text)
        if text_len > self.captcha_length:
            raise ValueError("验证码超过{}位啦！".format(self.captcha_length))
        for i in range(text_len):
            vector[i,self.captcha_list.index(text[i])] = 1    
        return vector

    # 验证码向量转为文本
    def vec2text(self,vec):
        label = torch.nn.functional.softmax(vec, dim =1)
        vec = torch.argmax(label, dim=1)
        for v in vec:
            text_list = [self.captcha_list[v] for v in vec]
        return ''.join(text_list)

    def make_dataset(self):
        img_names = os.listdir(self.data_path)
        img = []
        img_label = {}

        for img_name in img_names:
            tmp_type = img_name.split('.')[1]
            if(tmp_type == 'jpg' or tmp_type == 'jpeg' or tmp_type == 'png') :
                img.append(img_name)
            if(tmp_type == 'csv') :
                with open(os.path.join(self.data_path, img_name)) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        try:
                            tmp = row[0].split('.')[0]
                            if ((tmp!='ID')):
                                img_label[row[0]] = row[1]
                        except:
                            continue
        return img, img_label
    def calculat_acc(self, output, target):
        output, target = output.view(-1, len(self.captcha_list)), target.view(-1, len(self.captcha_list)) # 每37个就是一个字符
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output, target = output.view(-1, self.captcha_length), target.view(-1, self.captcha_length) #每6个字符是一个验证码
        c = 0
        for i, j in zip(target, output):
            if torch.equal(i, j):
                c += 1
        acc = c / output.size()[0] * 100
        return acc