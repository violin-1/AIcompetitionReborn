import torch.nn as nn
from base.BaseModule import BaseModel

class Net(BaseModel):
    def __init__(self,num_classes=10,model_path='.'):
        super(Net,self).__init__(model_path,'Alex')
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),#修改了这个地方，不知道为什么就对了
             # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256*5*20,4096),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Linear(4096,4*num_classes)
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x=self.features(x)
        # x=x.view(x.size(0),256*5*20)#256*1*1)
        # x=self.classifier(x)
        #return F.log_softmax(inputs, dim=3)
        return x