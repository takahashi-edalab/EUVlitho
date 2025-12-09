import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from torchvision.transforms import functional as Fv
from compmask import compmask
import time

ncut=1901
ncuts=1749
ncut2=2*ncut
ncuts2=2*ncuts

class CNNmodel(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1,16,(3,3),padding=1,padding_mode='circular')
		self.conv2 = nn.Conv2d(16,32,(3,3),padding=1,padding_mode='circular')
		self.conv3 = nn.Conv2d(32,64,(3,3),padding=1,padding_mode='circular')
		self.conv4 = nn.Conv2d(64,128,(3,3),padding=1,padding_mode='circular')
		self.conv5 = nn.Conv2d(128,256,(3,3),padding=1,padding_mode='circular')
		#self.conv6 = nn.Conv2d(512,1024,(3,3),padding=1,padding_mode='circular')
		self.norm1=nn.BatchNorm2d(16)
		self.norm2=nn.BatchNorm2d(32)
		self.norm3=nn.BatchNorm2d(64)
		self.norm4=nn.BatchNorm2d(128)
		self.norm5=nn.BatchNorm2d(256)
		#self.norm6=nn.BatchNorm2d(1024)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16*16*256, 4096)
		self.fc2=nn.Linear(4096, ncut)
	def forward(self, x):
		x=x.to(torch.float32)
		x = self.pool(F.relu(self.conv1(x)))
		x=self.norm1(x)
		x = self.pool(F.relu(self.conv2(x)))
		x=self.norm2(x)
		x = self.pool(F.relu(self.conv3(x)))
		x=self.norm3(x)
		x = self.pool(F.relu(self.conv4(x)))
		x=self.norm4(x)
		x = self.pool(F.relu(self.conv5(x)))
		x=self.norm5(x)
		#x = self.pool(F.relu(self.conv6(x)))
		#x=self.norm6(x)
		x=torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		outputs = self.fc2(x)
		return outputs

class CNNmodels(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1,16,(3,3),padding=1,padding_mode='circular')
		self.conv2 = nn.Conv2d(16,32,(3,3),padding=1,padding_mode='circular')
		self.conv3 = nn.Conv2d(32,64,(3,3),padding=1,padding_mode='circular')
		self.conv4 = nn.Conv2d(64,128,(3,3),padding=1,padding_mode='circular')
		self.conv5 = nn.Conv2d(128,256,(3,3),padding=1,padding_mode='circular')
		#self.conv6 = nn.Conv2d(512,1024,(3,3),padding=1,padding_mode='circular')
		self.norm1=nn.BatchNorm2d(16)
		self.norm2=nn.BatchNorm2d(32)
		self.norm3=nn.BatchNorm2d(64)
		self.norm4=nn.BatchNorm2d(128)
		self.norm5=nn.BatchNorm2d(256)
		#self.norm6=nn.BatchNorm2d(1024)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16*16*256, 4096)
		self.fc2=nn.Linear(4096, ncuts)
	def forward(self, x):
		x=x.to(torch.float32)
		x = self.pool(F.relu(self.conv1(x)))
		x=self.norm1(x)
		x = self.pool(F.relu(self.conv2(x)))
		x=self.norm2(x)
		x = self.pool(F.relu(self.conv3(x)))
		x=self.norm3(x)
		x = self.pool(F.relu(self.conv4(x)))
		x=self.norm4(x)
		x = self.pool(F.relu(self.conv5(x)))
		x=self.norm5(x)
		#x = self.pool(F.relu(self.conv6(x)))
		#x=self.norm6(x)
		x=torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		outputs = self.fc2(x)
		return outputs

class Fullmodel(nn.Module):
    def __init__(self,modelre0,modelim0,modelrex,modelimx,modelrey,modelimy):
        super().__init__()
        self.modelre0 = modelre0
        self.modelim0 = modelim0
        self.modelrex = modelrex
        self.modelimx = modelimx
        self.modelrey = modelrey
        self.modelimy = modelimy
    def forward(self, x):
        x1 = self.modelre0(x)
        x2 = self.modelim0(x)
        x3 = self.modelrex(x)
        x4 = self.modelimx(x)
        x5 = self.modelrey(x)
        x6 = self.modelimy(x)
        return x1,x2,x3,x4,x5,x6

def predict_m3d(mask,modelre0,modelim0,modelrex,modelimx,modelrey,modelimy,
    factor0,factorx,factory):
    fullmodel = Fullmodel(modelre0,modelim0,modelrex,modelimx,modelrey,modelimy)
    fullmodel.eval()
    x1,x2,x3,x4,x5,x6 = fullmodel(mask)

    re0=x1.detach().numpy()/factor0
    im0=x2.detach().numpy()/factor0
    rex=x3.detach().numpy()/factorx
    imx=x4.detach().numpy()/factorx
    rey=x5.detach().numpy()/factory
    imy=x6.detach().numpy()/factory
    return re0,im0,rex,imx,rey,imy

if __name__ == "__main__":
    modelre0=CNNmodel.load_from_checkpoint('../model/re0/model.ckpt')
    modelim0=CNNmodel.load_from_checkpoint('../model/im0/model.ckpt')
    modelrex=CNNmodels.load_from_checkpoint('../model/rex/model.ckpt')
    modelimx=CNNmodels.load_from_checkpoint('../model/imx/model.ckpt')
    modelrey=CNNmodels.load_from_checkpoint('../model/rey/model.ckpt')
    modelimy=CNNmodels.load_from_checkpoint('../model/imy/model.ckpt')
    factor0=np.loadtxt('../model/re0/factor0.csv',delimiter=',')
    factorx=np.loadtxt('../model/rex/factorx.csv',delimiter=',')
    factory=np.loadtxt('../model/rey/factory.csv',delimiter=',')

    maskname="mask.bin"
    nmask=512
    mask_test=compmask(maskname)
    mask_test=mask_test.reshape(-1,1,nmask,nmask)
    mask_test=torch.from_numpy(mask_test)

    start = time.perf_counter()
    re0,im0,rex,imx,rey,imy = predict_m3d(mask_test,modelre0,modelim0,modelrex,modelimx,modelrey,modelimy,
    factor0,factorx,factory)
    end = time.perf_counter()
#    print(f"Predict: {end-start: 5f}s")

    with open("inputxx.csv","w") as f:
        np.savetxt(f,re0, delimiter=",")
        np.savetxt(f,im0, delimiter=",")
        np.savetxt(f,rex, delimiter=",")
        np.savetxt(f,imx, delimiter=",")
        np.savetxt(f,rey, delimiter=",")
        np.savetxt(f,imy, delimiter=",")

