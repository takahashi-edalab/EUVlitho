import numpy as np

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

NDIV=2048
NDIVM=512

def compmask(filename: str):
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
	mask=np.empty((NDIVM,NDIVM),dtype='float16')
	row=read_bits(filename)
	mask0=np.empty(NDIV*NDIV,dtype='float16')
	for i in range(NDIV*NDIV):
    		mask0[i]=float(row[i])
	mask0=mask0.reshape(NDIV,NDIV)
	maskinput=np.dot(rowmat,np.dot(mask0,columnmat))
	return maskinput




