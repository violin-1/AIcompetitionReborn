import torch.nn as nn
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import dataloader
from model import loadnet
import os
import csv
from sklearn.model_selection import KFold
import numpy as np

class NetTrain():
    def __init__(self,batch_size = 64, learning_rate = 0.001):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.outcsv = []
        self.headers = ['cv_acc','test_acc','running_loss']

    def save_net(self,filename = None):
        path = os.path.join(self.model_path,'trainer',self.model_name+'.pth')
        # save model
        torch.save({
                'best_acc':self.best_acc,
                'optimizer_state_dict':self.optimizer.state_dict(),
                },path)

    def load_net(self):
        # 加载模型
        path = os.path.join(self.model_path,'trainer',self.model_name+'.pth')
        if os.path.exists(path):
            # print('开始加载模型')
            checkpoint = torch.load(path)
            self.best_acc = checkpoint['best_acc']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def set_net(self, model_path = None, model_name = None, loadbest = False):
        if(model_path != None):
            self.model_path = model_path
        if(model_name != None):
            self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = getattr(loadnet,self.model_name)( self.model_path, loadbest)
        self.net = self.net.to(self.device)
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def csv_pre_read(self):
        self.outcsv = []
        csv_path = os.path.join(self.model_path,'trainer')
        csv_name = self.model_name+'Result.csv'
        if os.path.exists(os.path.join(csv_path,csv_name)):
            with open(os.path.join(csv_path,csv_name)) as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        tmp = row[0]
                        if ((tmp!=self.headers[0])):
                            self.outcsv.append([float(row[0]),float(row[1]),float(row[2])])
                    except:
                        continue
        pass
    
    def show_csv(self, l = 0):
        
        outplt = list(map(list, zip(*self.outcsv)))
        l = len(self.outcsv)
        csv_path = os.path.join(self.model_path,'trainer')
        csv_name = self.model_name+'Result.csv'
        self.res_csv(csv_path,csv_name,self.outcsv,self.headers)
        x = list(range(0,l))
        if(len(outplt)):
            plt.figure()
            plt.plot(x,outplt[0],'r--',x,outplt[1],'b--',x,outplt[2],'g--')
            plt.legend(self.headers)
            plt.title(self.model_name)
            plt.show()
        pass

    def train(self,epoch_nums,setpath):

        self.csv_pre_read()
        train_dataset = dataloader.CaptchaDataSet.CaptchaData(os.path.join(setpath,'train'))
        kf = KFold(n_splits = epoch_nums, random_state = 4, shuffle = True)
        self.best_acc = 0
        self.load_net()

        for epoch,(train_index,test_index) in enumerate(kf.split(train_dataset)):

            traindataset = train_dataset.get_img(train_index)
            testdataset = train_dataset.get_img(test_index)
            trainloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=self.batch_size, shuffle=True)
            pre_epoch_total_step = len(trainloader)
            running_loss = 0

            self.net.train()
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                # forward
                prediction = self.net(x)
                loss = self.criterion(prediction, y)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % 100 == 0:
                    template = r"Epoch:{}/{}, step:{}/{}, Loss:{:.6f}"
                    print(template.format(epoch+1, epoch_nums, i+1, pre_epoch_total_step, loss.item()))
                running_loss += loss.item()

            c_acc = self.test(testdataset)
            t_acc = self.test(setpath=setpath)
            r_loss = running_loss
            self.outcsv.append([c_acc,t_acc,r_loss])
            print('第%s次训练的测试集正确率: %.3f %%, loss: %.3f' % (epoch+1,t_acc,running_loss))

            if epoch % 2 == 4:
                for p in self.optimizer.param_groups:
                    p['lr'] *= 0.9

            if(t_acc > self.best_acc):
                self.best_acc = t_acc
                self.net.save_net(True)
            self.net.save_net()
            self.save_net()
        self.show_csv(epoch_nums)
        
        pass

    def test(self,testdataset=None, setpath=None):

        if(testdataset == None and setpath != None):
            testdataset = dataloader.CaptchaDataSet.CaptchaData(setpath+'\\test')

        testloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=self.batch_size, shuffle=False)
        dataset = dataloader.CaptchaDataSet.CaptchaData(premode=0)
        totalacc = []
        # test model
        self.net.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs,labels = inputs.to(self.device),labels.to(self.device)
                outputs = self.net(inputs)
                acc = dataset.calculat_acc(outputs, labels)
                totalacc.append(acc)
        return np.mean(totalacc)
        
    def predict(self,setpath,outpath = None):
        test_data = dataloader.CaptchaDataSet.CaptchaData(setpath+'\\predict')
        test_data_loader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
        
        i = 1
        outputs = []
        self.net.eval()
        with torch.no_grad():
            for img, image_name in test_data_loader:
                img = img.to(self.device)
                out = self.net(img)
                out = out.view(-1, len(test_data.captcha_list))
                out = test_data.vec2text(out)
                tmp = [int(image_name[0].split('.')[0]),out]
                outputs.append(tmp)
                i += 1
                pass
        if(outpath == None):
            self.res_csv(self.model_path,self.model_name+'Submission.csv',outputs,['ID','label'])
        else:
            self.res_csv(outpath,self.model_name+'Submission.csv',outputs,['ID','label'])
        
    def res_csv(self,csv_path = None,csv_name = None,outputs = None,headers = None):
        if(csv_path != None and csv_name != None):
            with open(os.path.join(csv_path,csv_name),'w',newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(outputs)