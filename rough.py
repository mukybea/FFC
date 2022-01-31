import os 
import shutil
import math
import torch
import models
from dataset import train_test_loader

def get_device():
	    if torch.cuda.is_available(): 
	    	device = 'cuda:0' 
	    else: 
	    	device = 'cpu'
	    return device

device = get_device()
# print(os.listdir(os.path.join(os.getcwd(), "datasets/Train/view_1/View1_photo")))

# x = torch.randn(8,2,3,28,28).to(device)

# d = models.Builds().to(device)

# cd = d(x)

# print(cd.shape)

train_loader = train_test_loader.train_loader()
batch_size = next(iter(train_loader))[0].size(0)
print("batch_size", batch_size)
for a,b,c in train_loader:
	print(a.shape, b.shape, c.shape)