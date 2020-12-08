import torch.nn as nn
import torch
import os

class BaseModel(nn.Module):
    def __init__(self,model_path = '.',model_name = None):
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self.model_name = model_name

    def save_net(self,savebest = False):
        # path = self.model_path+'\\model\\'+self.model_name+'.pth'
        if(savebest):
            path = os.path.join(self.model_path, 'model', 'Best'+self.model_name+'.pth')
        else:
            path = os.path.join(self.model_path, 'model', self.model_name+'.pth')
        # save model
        torch.save(self.state_dict(),path)
        