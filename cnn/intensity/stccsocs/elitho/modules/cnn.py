import numpy as np
import csv
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from torchvision.transforms import functional as Fv

# ntest=1
ncut = 1901
ncuts = 1749
ncut2 = 2 * ncut
ncuts2 = 2 * ncuts
nmask = 512

mask_test = np.load("maskinput.npy")
mask_test = mask_test.reshape(-1, 1, nmask, nmask)
mask_test = torch.from_numpy(mask_test)


class CNNmodel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1, padding_mode="circular")
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1, padding_mode="circular")
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=1, padding_mode="circular")
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1, padding_mode="circular")
        # self.conv6 = nn.Conv2d(512,1024,(3,3),padding=1,padding_mode='circular')
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(128)
        self.norm5 = nn.BatchNorm2d(256)
        # self.norm6=nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 256, 4096)
        self.fc2 = nn.Linear(4096, ncut)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.norm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.norm3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.norm4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.norm5(x)
        # x = self.pool(F.relu(self.conv6(x)))
        # x=self.norm6(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        outputs = self.fc2(x)
        return outputs


class CNNmodels(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1, padding_mode="circular")
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1, padding_mode="circular")
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=1, padding_mode="circular")
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1, padding_mode="circular")
        # self.conv6 = nn.Conv2d(512,1024,(3,3),padding=1,padding_mode='circular')
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(128)
        self.norm5 = nn.BatchNorm2d(256)
        # self.norm6=nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 256, 4096)
        self.fc2 = nn.Linear(4096, ncuts)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.norm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.norm3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.norm4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.norm5(x)
        # x = self.pool(F.relu(self.conv6(x)))
        # x=self.norm6(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        outputs = self.fc2(x)
        return outputs


class Fullmodel(nn.Module):
    def __init__(self, modelre0, modelim0, modelrex, modelimx, modelrey, modelimy):
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
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        return x


modelre0 = CNNmodel.load_from_checkpoint(
    "/home/tanabe/emsim/SPIE2024/model/cnn/re0/model.ckpt"
)
modelim0 = CNNmodel.load_from_checkpoint(
    "/home/tanabe/emsim/SPIE2024/model/cnn/im0/model.ckpt"
)
modelrex = CNNmodels.load_from_checkpoint(
    "/home/tanabe/emsim/SPIE2024/model/cnn/rex/model.ckpt"
)
modelimx = CNNmodels.load_from_checkpoint(
    "/home/tanabe/emsim/SPIE2024/model/cnn/imx/model.ckpt"
)
modelrey = CNNmodels.load_from_checkpoint(
    "/home/tanabe/emsim/SPIE2024/model/cnn/rey/model.ckpt"
)
modelimy = CNNmodels.load_from_checkpoint(
    "/home/tanabe/emsim/SPIE2024/model/cnn/imy/model.ckpt"
)
model = Fullmodel(modelre0, modelim0, modelrex, modelimx, modelrey, modelimy)
# torch.save(model.state_dict(), 'model.pt')
# model.save_checkpoint("model.ckpt")

model.eval()
predict_test = model(mask_test).detach().numpy()
predict_test = np.transpose(predict_test)
# fpredict=open('./predict.csv','w')
# predictwriter=csv.writer(fpredict)
# predictwriter.writerows(predict_test)
# fpredict.close()
fpredict = open("./re0predict.csv", "w")
predictwriter = csv.writer(fpredict)
for source in range(0, ncut):
    predict = predict_test[source].reshape(-1)
    predictwriter.writerow(predict)
fpredict.close()
fpredict = open("./im0predict.csv", "w")
predictwriter = csv.writer(fpredict)
for source in range(ncut, ncut2):
    predict = predict_test[source].reshape(-1)
    predictwriter.writerow(predict)
fpredict.close()
fpredict = open("./rexpredict.csv", "w")
predictwriter = csv.writer(fpredict)
for source in range(ncut2, ncut2 + ncuts):
    predict = predict_test[source].reshape(-1)
    predictwriter.writerow(predict)
fpredict.close()
fpredict = open("./imxpredict.csv", "w")
predictwriter = csv.writer(fpredict)
for source in range(ncut2 + ncuts, ncut2 + ncuts2):
    predict = predict_test[source].reshape(-1)
    predictwriter.writerow(predict)
fpredict.close()
fpredict = open("./reypredict.csv", "w")
predictwriter = csv.writer(fpredict)
for source in range(ncut2 + ncuts2, ncut2 + ncuts2 + ncuts):
    predict = predict_test[source].reshape(-1)
    predictwriter.writerow(predict)
fpredict.close()
fpredict = open("./imypredict.csv", "w")
predictwriter = csv.writer(fpredict)
for source in range(ncut2 + ncuts2 + ncuts, ncut2 + ncuts2 + ncuts2):
    predict = predict_test[source].reshape(-1)
    predictwriter.writerow(predict)
fpredict.close()
