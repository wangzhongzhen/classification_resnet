
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet18_2c


# In[7]:


#device = 'cuda' if torch.cuda.is_available() else 'cpu'



# In[8]:


net = resnet18_2c.ResNet18()
net = torch.nn.DataParallel(net)    
#print('pretrain resnet is loading..')
#my_checkpoint = torch.load('my_resnet.t7')
#my_checkpoint = torch.load('my_ckpt180.t7')
#pretrain = my_checkpoint['net']
#net.load_state_dict(pretrain)


class My_net(nn.Module):
    def __init__(self,net):
        super(My_net,self).__init__()
        self.net = net
        
    def forward(self,x):
        #print(x0.shape)
        y0c,y0s = self.net(x)
        y0c = y0c.view((int(y0c.shape[0]/4),4,2))
        y0s = y0s.view((int(y0s.shape[0]/4),4,2))
        pc, idxc = torch.max(y0c, 1)
        ps, idxs = torch.max(y0s, 1)
#         print(ps.shape)
        return pc, ps, idxc, idxs



def my_net():
    return My_net(net)


