import torch.nn as nn
import matplotlib.pyplot as plt
import os

import dataloader
import Trainer

def delfile(path):
    if(os.path.exists(path)):
        os.remove(path)

def delmodelfile(model_path,model_name):
    path = os.path.join(model_path,model_name+'Submission.csv')
    delfile(path)
    ndirs = ['model','trainer']
    for i,ndir in enumerate(ndirs):
        path = os.path.join(model_path,ndir,model_name+'.pth')
        delfile(path)
        if(i == 0):
            path = os.path.join(model_path,ndir,'Best'+model_name+'.pth')
            delfile(path)
        else:
            path = os.path.join(model_path,ndir,model_name+'Result.csv')
            delfile(path)

def main():
    Trainningway = Trainer.NetTrain(batch_size=10)
    setpath = '.'#os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    AllNetName = ['Captcha','ResNet','PspNet']
   
    fac = 1
    while(fac!='0'):
        tmp = input('0：CaptchaNet，1:ResNet，2：PspNet\n')
        if(tmp != ''):
            model_select =  int(tmp)
        model_path = input('模型所在路径（C:\\windows）')
        if(len(model_path) == 0):
            model_path = '.'

        select = input('0：训练数据，1:测试数据，2:预测结果, 3:查看当前结果, 4:清除历史数据\n')
        if(select == '0'):
            n = int(input('请输入迭代次数：'))
            Trainningway.set_net(model_path,AllNetName[model_select])
            Trainningway.train(n,setpath)
        elif(select == '1'):
            Trainningway.set_net(model_path,AllNetName[model_select],True)
            acc = Trainningway.test(setpath=setpath)
            print('测试集正确率：{}'.format(acc))
        elif(select == '2'):
            out_path = input('请输入结果存放路径')
            if(len(out_path) == 0):
                out_path = None
            Trainningway.set_net(model_path,AllNetName[model_select],True)
            Trainningway.predict(setpath,outpath = out_path)
        elif(select == '3'):
            for i in range(0,len(AllNetName)):
                Trainningway.set_net(model_path,AllNetName[i])
                Trainningway.csv_pre_read()
                Trainningway.show_csv()
        elif(select == '4'):
            for i in range(0,len(AllNetName)):
                delmodelfile(model_path,AllNetName[i])
        fac = input('按0退出按任意键继续')
        if(fac == ''):
            fac = 1

if __name__ == "__main__":
    main()
    pass