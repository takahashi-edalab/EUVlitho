import matplotlib.pyplot as plt
import numpy as np
import csv
from math import sqrt, pi, sin, cos
import os
import torch
from torch import nn
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision
from torchvision.transforms import functional as Fv
import pandas as pd

NDIVM=512

sinlm=np.empty(NDIVM,dtype='float32')
coslm=np.empty(NDIVM,dtype='float32')
for i in range(NDIVM):
    sinlm[i]=sin(2.*pi*i/NDIVM)
    coslm[i]=cos(2.*pi*i/NDIVM)

def rshift(ncut,lorder,morder,rea,ima,idx,lshift,mshift):
    ramp=np.empty(ncut,dtype='float32')
    for i in range(ncut):
        lm=(lorder[i]*lshift+morder[i]*mshift)%NDIVM
        ramp[i]=rea[idx][i]*coslm[lm]-ima[idx][i]*sinlm[lm]
    return ramp

def ishift(ncut,lorder,morder,rea,ima,idx,lshift,mshift):
    iamp=np.empty(ncut,dtype='float32')
    for i in range(ncut):
        lm=(lorder[i]*lshift+morder[i]*mshift)%NDIVM
        iamp[i]=rea[idx][i]*sinlm[lm]+ima[idx][i]*coslm[lm]
    return iamp

class MaskampDatasetTrain(Dataset):
    def __init__(self,ntrain,ndata,maskall,ncut,lorder,morder,rea,ima):
        self.ntrain=ntrain
        self.ndata=ndata
        self.maskall=maskall
        self.ncut=ncut
        self.lorder=lorder
        self.morder=morder
        self.rea=rea
        self.ima=ima
    def __len__(self):
        return self.ntrain
    def __getitem__(self, idx):
        id=idx%self.ndata
        lshift=random.randrange(NDIVM)
        mshift=random.randrange(NDIVM)
        x=self.maskall[id]
        x=np.roll(x,(-lshift,-mshift),axis=(0,1))
        y=rshift(self.ncut,self.lorder,self.morder,self.rea,self.ima,id,lshift,mshift)
#        y=ishift(self.ncut,self.lorder,self.morder,self.rea,self.ima,id,lshift,mshift)
        x=x.reshape(1,NDIVM,NDIVM)
        x=torch.from_numpy(x)
        y=torch.from_numpy(y)
        return x, y

class MaskampDatasetVal(Dataset):
    def __init__(self,nval,maskall,ncut,lorder,morder,rea,ima):
        self.nval=nval
        self.maskall=maskall
        self.ncut=ncut
        self.lorder=lorder
        self.morder=morder
        self.rea=rea
        self.ima=ima
    def __len__(self):
        return self.nval
    def __getitem__(self, idx):
        x=self.maskall[idx]
        y=rshift(self.ncut,self.lorder,self.morder,self.rea,self.ima,idx,0,0)
#        y=ishift(self.ncut,self.lorder,self.morder,self.rea,self.ima,idx,0,0)
        x=x.reshape(1,NDIVM,NDIVM)
        x=torch.from_numpy(x)
        y=torch.from_numpy(y)
        return x, y

class CNNmodel(pl.LightningModule):
	def __init__(self,ncutcnn):
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
		self.fc2=nn.Linear(4096, ncutcnn)
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
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(),lr=0.001,eps=0.001)
#		optimizer = torch.optim.Adam(self.parameters(),lr=0.001,eps=0.0001)
#		optimizer = torch.optim.SGD(self.parameters(),lr=0.05,momentum=0.)
		return optimizer
	def training_step(self, train_batch, batch_idx):
		mask, targets= train_batch
		preds = self.forward(mask)
		loss = F.mse_loss(preds, targets)	
		self.log('train_loss',loss,on_step=False,on_epoch=True,sync_dist=True)
		return loss
	def validation_step(self, val_batch, batch_idx):
		mask, targets = val_batch
		preds = self.forward(mask)
		loss = F.mse_loss(preds, targets)
		self.log('val_loss', loss,sync_dist=True)
		for i in range(len(preds[0,:])):
			loss = F.mse_loss(preds[:,i], targets[:,i])
			self.log('val_loss_'+str(i), loss, sync_dist=True)

if __name__ == "__main__":
    ntrain=100000
    ndata=2000
    nval=200

    ncut=1901
#    ncut=1749

    traindir='../../data/train/'
    valdir='../../data/validate/'

    lorder=np.load(traindir+'inputlm/lorder.npy')
    morder=np.load(traindir+'inputlm/morder.npy')
#    lorder=np.load(traindir+'inputlm/lorders.npy')
#    morder=np.load(traindir+'inputlm/morders.npy')

    rea_train=np.load(traindir+'ampdata/re0.npy')
    ima_train=np.load(traindir+'ampdata/im0.npy')
    fac=np.empty(ncut,dtype='float32')
    for i in range(ncut):
        sum=0.
        for idx in range(ndata):
            ramp=rea_train[idx][i]
            iamp=ima_train[idx][i]
            sum+=(ramp*ramp+iamp*iamp)/2.
        fac[i]=1./sqrt(sum/ndata)
    with open('factor0.csv','w') as ffactor:
        facwriter=csv.writer(ffactor)
        facwriter.writerow(fac)
    for idx in range(ndata):
        for i in range(ncut):
            rea_train[idx][i]=rea_train[idx][i]*fac[i]
            ima_train[idx][i]=ima_train[idx][i]*fac[i]

    rea_val=np.load(valdir+'ampdata/re0.npy')
    ima_val=np.load(valdir+'ampdata/im0.npy')
    for idx in range(nval):
        for i in range(ncut):
            rea_val[idx][i]=rea_val[idx][i]*fac[i]
            ima_val[idx][i]=ima_val[idx][i]*fac[i]

    maskall_train=np.load(traindir+'maskdata/mask.npy')
    maskall_val=np.load(valdir+'maskdata/mask.npy')

    data_train = MaskampDatasetTrain(ntrain,ndata,maskall_train,ncut,lorder,morder,rea_train,ima_train)
    data_val = MaskampDatasetVal(nval,maskall_val,ncut,lorder,morder,rea_val,ima_val)

#    train_dataloader=DataLoader(data_train,batch_size=128,shuffle=True,num_workers=os.cpu_count(),pin_memory=True)
#    val_dataloader=DataLoader(data_val,batch_size=128,num_workers=os.cpu_count(),pin_memory=True)
    train_dataloader=DataLoader(data_train,batch_size=128,shuffle=True,num_workers=4*torch.cuda.device_count(),pin_memory=True)
    val_dataloader=DataLoader(data_val,batch_size=128,num_workers=4*torch.cuda.device_count(),pin_memory=True)

    model = CNNmodel(ncut)
    logger = pl.loggers.CSVLogger("./",version=0, name="history")
    trainer = pl.Trainer(accelerator='gpu',devices=[0],strategy="ddp",logger=logger,max_epochs=50)
    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint("model.ckpt")
#    model = model.load_from_checkpoint("model.ckpt")

    log=pd.read_csv('./history/version_0/metrics.csv')
    log=log.groupby("epoch").max()
    log.to_csv('./log.csv')

    loss_values=log['train_loss']
    val_loss_values=log['val_loss']
    epochs=range(1,len(loss_values)+1)
    plt.rcParams.update({'font.size': 16})
    plt.plot(epochs,loss_values,'bo',label='Training loss')
    plt.plot(epochs,val_loss_values,'r',label='Validation loss')
    plt.ylim(bottom=0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./out.png')
    plt.show()

