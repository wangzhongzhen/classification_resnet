
# coding: utf-8

# In[256]:


import torch
print(torch.__version__)
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import resnet_max_p_
import torchvision.transforms as transform
import os
import time
import numpy as np
import cv2
from PIL import Image
from torch.autograd import Variable


# In[257]:


# model_path = 'maxp.t7'
# dst_folder = './P00012233T20180622R084430'


# In[258]:


class Cpr_classification:
    
    def __init__(self,dst_folder):
        self.pretrain = 'maxp.t7'
        self.dst_folder = dst_folder
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([0.5],[0.5])
        ])
        
        model = resnet_max_p_.my_net()
        
        checkpoint = torch.load(self.pretrain)  #
        #print(checkpoint['acc_c'])
        model.load_state_dict(checkpoint['net'])
        
        if torch.cuda.is_available():
            model.cuda()
        self.model = model.eval()
        
   
        
    def forward_data(self):
        B_TABLE = {
        'pRCA' : 'RCA',
        'mRCA' : 'RCA',
        'dRCA' : 'RCA',
        'R-PDA' : 'R-PDA',
        'R-PLB' : 'R-PLB',
        'RI' : 'RI',
        'LM' : 'LAD',
        'pLAD' : 'LAD',
        'mLAD' : 'LAD',
        'dLAD' : 'LAD',
        'D1' : 'D1',
        'D2' : 'D2',
        'pCx' : 'LCX',
        'LCx' : 'LCX',
        'OM1' : 'OM1',
        'OM2' : 'OM2',
                      }
        NAME_TABLE = [
        '0000',
        '0009',
        '0018',
        '0028',
        '0037',
        '0047',
        '0056',
        '0066',
        '0075',
        '0085',
        '0094',
        '0104',
        '0113',
        '0123',
        '0132',
        '0142',
        '0151',
        '0161',
        '0170',
        '0180'
                  ]
        print(self.dst_folder)
        csv_file = os.path.join(self.dst_folder,'narrow_list', 'narrow_result_classified.csv')
        #print(csv_file)
        if False == os.path.exists(csv_file):
            return
        fp = open(os.path.join(self.dst_folder,'narrow_list', 'narrow_result_cpr_classified.csv'), 'w')
        
        for in_line in open(csv_file):
            in_line = in_line[:-1]
            
            key = in_line.split(',')[1]
            in_vessel_name =  B_TABLE[key]
            
            narrow_center = in_line.split(',')[4]
            narrow_type = int(in_line.split(',')[5])
            narrow_center = int(narrow_center) - 65
            
            img = np.zeros((4,60,60))            
            for j,i in enumerate([0,5,10,15]):
                           
                #print(in_vessel_name)
                strnum = NAME_TABLE[i]
                coords_txt = os.path.join(self.dst_folder,'cpr',in_vessel_name,strnum+'.txt')
                
                
                imgfile = os.path.join(self.dst_folder,'cpr',in_vessel_name,'noline/'+strnum+'.png')
#                 print(imgfile)
               
                coords = open(coords_txt).readlines()
                data = coords[narrow_center].strip().split(' ')
                
                x, y = int(data[0]),int(data[1])
                
                x0 = x - 30
                x1 = x + 30
                y0 = y - 30
                y1 = y + 30
                if x0 < 0:
                    x0 = 0
                    x1 = 60
                if x1 > 511:
                    x0 = 511-60
                    x1 = 511
                if y0 < 0:
                    y0 = 0
                    y1 = 60
                if y1 > 511:
                    y0 = 511-60
                    y1 = 511
                
                
                rawimg = cv2.imread(imgfile)
                
                rawimg = rawimg[:,:,0].T
                cutimg = rawimg[x0:x1,y0:y1]
                
                img[j] = self.transform(Image.fromarray(cutimg).convert('L'))
            
            img = torch.from_numpy(img).unsqueeze(0)

            x = img.float().view((img.shape[0]*4,1)+img.shape[2:])
            input = Variable(x)
            if torch.cuda.is_available():
                input = input.cuda()
            p_c, p_s,_,__ = self.model(input)
            _,predict_c = p_c.max(1)
            _,predict_s = p_s.max(1)

            predict_c = int(predict_c.cpu().data.numpy())
            predict_s = int(predict_s.cpu().data.numpy())
            #print(predict_c,predict_s)
            if predict_c == 1 and predict_s== 1:
                label = 0
            elif predict_c == 0 and predict_s == 1:
                label = 1
            elif predict_c == 0 and predict_s == 0:
                label = 3
            elif predict_c == 1 and predict_s == 0:
                label = 2
           
            fp.write(in_line+','+str(label) +'\n')
        print('write to narrow_result_cpr_classified.csv')
        fp.close()


# In[259]:


# cpr = Cpr_classification(model_path,dst_folder)
# cpr.forward_data()

