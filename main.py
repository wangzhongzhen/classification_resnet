
# coding: utf-8

# In[1]:


import os,sys
import cpr_classification


# In[2]:


if len(sys.argv) < 2:
    print('dst_folder')
    
dst_folder = sys.argv[1]
# dst_folder = './P00012233T20180622R084430'
net = cpr_classification.Cpr_classification(dst_folder=dst_folder)
net.forward_data()

