import numpy as np
import csv
import os

def read_bits(filename):
    bits = []
    with open(filename, "rb") as f:
        byte = f.read(1)
        while byte:
            b = byte[0]
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
            byte = f.read(1)
    return bits

ndata=2000

NDIV=2048
NDIVM=512
ncut=1901
ncuts=1749

path_list=['./maskdata','./ampdata','./inputlm']
for path in path_list:
    os.makedirs(path, exist_ok=True)

rowmat=np.zeros((NDIVM,NDIV),dtype='float16')
bin_size = NDIV/NDIVM
next_bin_break = bin_size
which_row = 0
which_column = 0
while ((which_row <NDIVM)&(which_column < NDIV)):
    if ((next_bin_break - which_column) >= 1):
        rowmat[which_row, which_column] = 1/bin_size
        which_column += 1
    elif (abs(next_bin_break - which_column)<0.0000001):
        which_row += 1
        next_bin_break += bin_size
columnmat=np.empty((NDIV, NDIVM),dtype='float16')
for i in range(NDIV):
    for j in range(NDIVM):
        columnmat[i,j]=rowmat[j,i]

maskall=np.empty((ndata,NDIVM,NDIVM),dtype='float16')
row=read_bits('mask.bin')
for idx in range(ndata):
    mask0=np.empty(NDIV*NDIV,dtype=int)
    for i in range(NDIV*NDIV):
        mask0[i]=int(row[i+idx*NDIV*NDIV])
    mask0=mask0.reshape(NDIV,NDIV)
    mask=np.dot(rowmat,np.dot(mask0,columnmat))
    maskall[idx]=mask
np.save('./maskdata/mask.npy',maskall)

re0=np.empty((ndata,ncut),dtype='float32')
im0=np.empty((ndata,ncut),dtype='float32')
rex=np.empty((ndata,ncuts),dtype='float32')
imx=np.empty((ndata,ncuts),dtype='float32')
rey=np.empty((ndata,ncuts),dtype='float32')
imy=np.empty((ndata,ncuts),dtype='float32')
with open('inputxx.csv',newline='') as finputxx:
    freader=csv.reader(finputxx)
    for idx in range(ndata):
        row=next(freader)
        for i in range(ncut):
            re0[idx][i]=float(row[i])
        row=next(freader)
        for i in range(ncut):
            im0[idx][i]=float(row[i])
        row=next(freader)
        for i in range(ncuts):
            rex[idx][i]=float(row[i])
        row=next(freader)
        for i in range(ncuts):
            imx[idx][i]=float(row[i])
        row=next(freader)
        for i in range(ncuts):
            rey[idx][i]=float(row[i])
        row=next(freader)
        for i in range(ncuts):
            imy[idx][i]=float(row[i])
np.save('./ampdata/re0.npy',re0)
np.save('./ampdata/im0.npy',im0)
np.save('./ampdata/rex.npy',rex)
np.save('./ampdata/imx.npy',imx)
np.save('./ampdata/rey.npy',rey)
np.save('./ampdata/imy.npy',imy)

with open('inputlm.csv',newline='') as finputlm:
    freader=csv.reader(finputlm)
    content=[[int(row[0]),int(row[1]),int(row[2])] for row in freader]
lorder=np.empty(ncut,dtype=int)
morder=np.empty(ncut,dtype=int)
xorder=np.empty(ncut,dtype=int)
for i in range(ncut):
    lorder[i]=content[i][0]
    morder[i]=content[i][1]
    xorder[i]=content[i][2]
lorders=np.empty(ncuts,dtype=int)
morders=np.empty(ncuts,dtype=int)
nc=0
for i in range(ncut):
    if xorder[i] == 1:
        lorders[nc]=lorder[i]
        morders[nc]=morder[i]
        nc=nc+1
np.save('./inputlm/lorder.npy',lorder)
np.save('./inputlm/morder.npy',morder)
np.save('./inputlm/lorders.npy',lorders)
np.save('./inputlm/morders.npy',morders)

nflip=np.empty(ncut,dtype=int)
nflips=np.empty(ncuts,dtype=int)
for i in range(ncut):
    l=lorder[i]
    m=morder[i]
    for j in range(ncut):
        if((lorder[j]==-l)&(morder[j]==m)):
            nflip[i]=j
for i in range(ncuts):
    l=lorders[i]
    m=morders[i]
    for j in range(ncuts):
        if((lorders[j]==-l)&(morders[j]==m)):
            nflips[i]=j
#for i in range(ncut):
#   print(nflip[i])
#for i in range(ncuts):
#    print(nflips[i])

"""
path_list=['./maskdata_wflip','./ampdata_wflip']
for path in path_list:
    os.makedirs(path, exist_ok=True)

fmaskall=np.empty((2*ndata,NDIVM,NDIVM),dtype='float16')
fre0=np.empty((2*ndata,ncut),dtype='float32')
fim0=np.empty((2*ndata,ncut),dtype='float32')
frex=np.empty((2*ndata,ncuts),dtype='float32')
fimx=np.empty((2*ndata,ncuts),dtype='float32')
frey=np.empty((2*ndata,ncuts),dtype='float32')
fimy=np.empty((2*ndata,ncuts),dtype='float32')
for idx in range(ndata):
    fmaskall[idx]=maskall[idx]
    fre0[idx]=re0[idx]
    fim0[idx]=im0[idx]
    frex[idx]=rex[idx]
    fimx[idx]=imx[idx]
    frey[idx]=rey[idx]
    fimy[idx]=imy[idx]
    mask=maskall[idx]
    fmaskall[ndata+idx]=np.flipud(mask)
    for i in range(ncut):
        fre0[ndata+idx][i]=re0[idx][nflip[i]]
        fim0[ndata+idx][i]=im0[idx][nflip[i]]
    for i in range(ncuts):
        frex[ndata+idx][i]=-rex[idx][nflips[i]]
        fimx[ndata+idx][i]=-imx[idx][nflips[i]]
    for i in range(ncuts):
        frey[ndata+idx][i]=rey[idx][nflips[i]]
        fimy[ndata+idx][i]=imy[idx][nflips[i]]

np.save('./maskdata_wflip/mask.npy',fmaskall)
np.save('./ampdata_wflip/re0.npy',fre0)
np.save('./ampdata_wflip/im0.npy',fim0)
np.save('./ampdata_wflip/rex.npy',frex)
np.save('./ampdata_wflip/imx.npy',fimx)
np.save('./ampdata_wflip/rey.npy',frey)
np.save('./ampdata_wflip/imy.npy',fimy)
"""



