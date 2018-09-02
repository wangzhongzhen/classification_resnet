
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pylab as plt
import random
from PIL import Image
import resnet_max_p_
import torch.optim as optim
import torch.nn.functional as F
import time


# In[20]:


train_data = 'train_data'
val_data = 'val_data'
j = 0
batchsize = 256
clr = 0
log1 = 'train_loss_2c'
log2 = 'val_loss_2c'
start_epoch = 0
decay = 0.1
epoch_decay = 50
n_epoch = 200
fix = 20
save_net = True
resume = False
best_loss = 2
# In[3]:


save_dir = 'loss'

my_checkpoint = 'my_checkpoint'

print('batchsize:',batchsize,'clr:',clr)
# In[4]:


my_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


# In[5]:


transform0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


# In[6]:


def D(list):
    # 去掉mac中的Ds
    if '.DS_Store' in list:
            list.remove('.DS_Store')
    return list


# In[7]:


def default_loader(path,phase):
    path_ = os.path.join(path,'cpr')
    cpr_list = D(sorted(os.listdir(path_)))
    num = len(cpr_list)
#     print(num)
#     for i in range(num):
#         print(cpr_list[i])
#     print('----')
    img = np.ones((4,60,60))
    
    if num==20:
        if phase == 'train':
        #start = random.randint(0,4)#torch
            start = int(torch.randint(0,4,(1,)))
        # print(start)
            for i in range(4):
                pic_cpr = cpr_list[start + i*5]
                pic_path = os.path.join(path_,pic_cpr)
    #             print(pic_path)
                img0 = cv2.imread(pic_path,0)

    #             img0 = Image.open(pic_path).convert('L')
                img[i] = img0
        else:
            for i in range(4):
                pic_cpr = cpr_list[0 + i*5]
                pic_path = os.path.join(path_,pic_cpr)
                #print(pic_path)
                img0 = cv2.imread(pic_path,0)
                img[i] = img0
                
    else:
        if phase == 'train':
            j = int(torch.randint(0,num,(1,)))
        else:
            j = 0
        #print(j)
        pic_path = os.path.join(path_,cpr_list[j])
        img_ = cv2.imread(pic_path,0)
        #             img_ = Image.open(pic_path).convert('L')
        img = np.concatenate((img_[np.newaxis,:,:],img_[np.newaxis,:,:],img_[np.newaxis,:,:],img_[np.newaxis,:,:]),axis=0)
        #             for i in range(4):
        #                 img[i] = img_
        #             print(img.shape)
#     print(img[3].shape)
#     plt.imshow(img[3])
#     plt.show()

    return img

# In[8]:


class myImageFloder(Dataset):
    def __init__(self, workspace, default_loader, my_transforms = None, phase='train'):
        self.workspace = workspace
        self.loader = default_loader
        self.my_transform = my_transforms
        self.phase = phase
        img_id = []
        
        class_ = os.listdir(workspace)
        # 
        for label in class_:
            if label[0]=='.':
                continue
            path = os.path.join(workspace,label)   #train_data/2
            
            if path[0]=='.':
                continue
            for id_ in os.listdir(path):
                if id_[0]=='.' or id_.endswith('png'):
                    continue
                id_path = os.path.join(path,id_)
#                 print(id_path)
                
                img_id.append(id_path)
                
        self.img_id = img_id
#         for i,j in enumerate(range(len(self.img_id))):
#             print(i,self.img_id[j])
              
            
    def __getitem__(self,index):
        #print(index)
        img_id = self.img_id[index]
        img_label = img_id.split('/')[1]
#         print(img_label)
        img_ = self.loader(img_id,self.phase)
        img = np.zeros((4,60,60))      
        if self.my_transform is not None:
            
            for i in range(4):
                img[i] = self.my_transform(Image.fromarray(img_[i]).convert('L'))
                
        else:
            for i in range(4):
                img[i] = transform0(Image.fromarray(img_[i]).convert('L'))
#         plt.imshow(img[0])
#         plt.show()
        return img,int(img_label)
    def __len__(self):
        return len(self.img_id)
    
# data_loader = myImageFloder(val_data,default_loader,my_transforms=my_transforms)
# for i,(img,label) in enumerate(data_loader):
#     print(img.shape)
#     plt.imshow(img[3])
#     plt.show()


# In[9]:


trainset = myImageFloder(train_data,default_loader,my_transforms,phase = 'train')
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True,num_workers=2)


# In[10]:


testset = myImageFloder(val_data,default_loader,phase = 'test')
testloader = torch.utils.data.DataLoader(testset,batch_size=8,shuffle=False,num_workers=2)


# In[11]:


def writeLossLog(phase,epoch, meanloss,meanpre,meanloss_s,meanpre_s,log,save_dir=save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, "%s.txt" % log)
        #    print ('save_dir: %s', save_dir)

        exp_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        fp = open(save_dir, 'a')
        fp.write(str(epoch) + ' ' + phase +' '+ 'loss_c'+' ' + str(round(meanloss,5)) +' '+'acc_c'+' '+ str(round(meanpre,5))+ ' ' + 'loss_s'+' ' + str(round(meanloss_s,5)) +' '+'acc_s'+' '+ str(round(meanpre_s,5))+' '+str(exp_id) + '\n')

        fp.close()
        return

def adjust_learning_rate(optimizer,lr,decay):
    lr *= decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# In[12]:


path0 = 'loss/train_loss_2c.txt'
path1 = 'loss/val_loss_2c.txt'
if os.path.exists(path0):
    os.remove(path0)
if os.path.exists(path1):    
    os.remove(path1)


# In[13]:


# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet_max_p_.my_net()
model.to(device)

criterion = nn.CrossEntropyLoss()

for children in model.children():
    for i,param in enumerate(children.parameters()):
        if i < fix:
            param.requires_grad = False


# In[14]:


optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()),lr=clr,momentum=0.99)


if resume == True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('my_checkpoint0'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./my_checkpoint0/maxp_ckpt300.t7')
    model.load_state_dict(checkpoint['net'])
    model.net = model.net.module
    #best_acc = checkpoint['acc']
    #start_epoch = checkpoint['epoch']




# In[15]:


def loss1(output1,targets):
    # calc or not
    target = (((targets==2) + (targets==0))>0).long()
#     target = [0 if x==1 or x==3 else 1 for x in targets]
#     target = torch.tensor(target)
    
#     target = target.to(device)
#     print (output1.shape,target)
    loss= criterion(output1,target)
    
    return loss,target

def loss2(output1,targets):
    target = (((targets==1) + (targets==0))>0).long()

    # soft or not
#     target = [0 if x==2 or x==3 else 1 for x in targets]
#     target = torch.tensor(target)
#     target = target.to(device)
    return criterion(output1,target),target


# In[16]:


def train(epoch):
    #print('train...',epoch)
    model.train()
    for m in model.net.modules():
    #    print(m)
        if isinstance(m,nn.BatchNorm2d):
    #   print('is bn')
     #if isinstance(m,nn.BatchNorm2d):
            m.momentum=0.01
    #        m.eval()
    #for i_p,(name,p) in enumerate(model.net.state_dict().items()):
    #    if 'bn' in name:
     #       print([name,p.data[:10]])
      #  if i_p >10:
       #     break
        
    train_loss = 0
    correct_c = 0
    total_c = 0
    metrics_main0 = []
    metrics_main1 = []
    
    correct_s = 0
    total_s = 0
    metrics_main0_ = []
    metrics_main1_ = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs.float().view((inputs.shape[0]*4,1)+inputs.shape[2:])

        p_c,p_s,idxc,idxs = model(x)
#         print(F.softmax(p_c),F.softmax(p_s))
        loss_c,target_c = loss1(p_c,targets)
        loss_s,target_s = loss2(p_s,targets)
        #print(loss_c,target_c)
        loss = loss_c + loss_s
#         print([loss_c, loss_s])

        loss.backward()
        optimizer.step()
        
        # cala
        _,predicted_c = p_c.max(1)
        total_c += target_c.size(0)
        correct_c += predicted_c.eq(target_c).sum().item()
        
        metrics_main0.append(loss_c.item())
        metrics_main1.append(100.*correct_c/total_c)
        

        # soft
        
        _, predicted_s = p_s.max(1)
        total_s += target_s.size(0)
        correct_s += predicted_s.eq(target_s).sum().item()
        
        metrics_main0_.append(loss_s.item())
        metrics_main1_.append(100.*correct_s/total_s)
        
    print('train','  loss_c: %.3f'% np.mean(metrics_main0),'   Acc_c: %0.3f'%np.mean(metrics_main1),
         '  loss_s: %.3f'% np.mean(metrics_main0_),'   Acc_s: %0.3f'%np.mean(metrics_main1_))
    writeLossLog('train',epoch,np.mean(metrics_main0),np.mean(metrics_main1),np.mean(metrics_main0_),
                                 np.mean(metrics_main1_),log1)
        
# train(1)      


# In[17]:


def test(epoch):
    global clr,best_loss
    #print('test...',epoch)
    model.eval()
    
    train_loss = 0
    correct_c = 0
    total_c = 0
    metrics_main0 = []
    metrics_main1 = []
    metrics_total_loss = []
    correct_s = 0
    total_s = 0
    metrics_main0_ = []
    metrics_main1_ = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        #optimizer.zero_grad()
        x = inputs.float().view((inputs.shape[0]*4,1)+inputs.shape[2:])

        p_c,p_s,idxc,idxs = model(x)
#         print(F.softmax(p_c),F.softmax(p_s))
        loss_c,target_c = loss1(p_c,targets)
        loss_s,target_s = loss2(p_s,targets)
# #         print(loss_s)
        loss = loss_c + loss_s
#         print([loss_c, loss_s])
#         loss.backward()
#         optimizer.step()
        
        # cala
        _,predicted_c = p_c.max(1)
        total_c += target_c.size(0)
        correct_c += predicted_c.eq(target_c).sum().item()
        
        metrics_main0.append(loss_c.item())
        metrics_main1.append(100.*correct_c/total_c)
        

        # soft
        
        _, predicted_s = p_s.max(1)
        total_s += target_s.size(0)
        correct_s += predicted_s.eq(target_s).sum().item()
        
        metrics_main0_.append(loss_s.item())
        metrics_main1_.append(100.*correct_s/total_s)
        metrics_total_loss.append(loss.item())
        
    print('test','  loss_c: %.3f'% np.mean(metrics_main0),'   Acc_c: %0.3f'%np.mean(metrics_main1),
         '  loss_s: %.3f'% np.mean(metrics_main0_),'   Acc_s: %0.3f'%np.mean(metrics_main1_))
    writeLossLog('test',epoch,np.mean(metrics_main0),np.mean(metrics_main1),np.mean(metrics_main0_),np.mean(metrics_main1_),log2)

    if epoch % epoch_decay == 0 and epoch!=0:
        clr = adjust_learning_rate(optimizer,lr=clr,decay = decay)
    
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        
    
    
    if save_net == True:
        
        print('loading loss:',np.mean(metrics_total_loss))
        #if best_loss > np.mean(metrics_total_loss):
        if epoch % 20 == 0 and epoch!=0:
            print('net saving..')
            state = {
                'net':model.state_dict(),
                'total_loss':np.mean(metrics_total_loss),
                'acc_c':np.mean(metrics_main1),
                'acc_s':np.mean(metrics_main1_)
            }
            if not os.path.exists(my_checkpoint):
                os.makedirs(my_checkpoint)
            torch.save(state,os.path.join(my_checkpoint,'maxp_ckpt%s.t7'%epoch))
            #torch.save(state,os.path.join(my_checkpoint,'maxp_ckpt.t7'))
            best_loss = np.mean(metrics_total_loss)

# In[18]:


# test(1)


# In[23]:


for epoch in range(start_epoch, n_epoch):
    print('Epoch:',epoch)
    train(epoch)
    test(epoch)

