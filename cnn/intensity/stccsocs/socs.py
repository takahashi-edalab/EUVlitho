import numpy as np
import cupy as cp
import csv
import pickle
import os
import bitarray
import time
from elitho import const,pupil, descriptors, diffraction_order
from reflectivity import reflect_amplitude
from eigenfunction import eigenfunction

def intsocs(cpattern, a0,ax,ay,phase0,
		alphaXs0,phiXs0,alphaXsx,phiXsx,alphaXsy,phiXsy):
	ncut=a0.shape[0];  # number of (l,m) pairs for a0

#	fpattern=np.fft.fft2(pattern)
#	start = time.perf_counter()
	cfpattern=cp.fft.fft2(cpattern)
#	end = time.perf_counter()
#	print(f"FFT: {end-start: 5f}s")
	fpattern=cfpattern.get()

	fmask = np.empty((const.noutX, const.noutY), dtype=complex)
	for i in range(const.noutX):
		l=i-const.lpmaxX
		for j in range(const.noutY):
			m=j-const.lpmaxY
			fmask[i,j]=fpattern[l,m]/const.NDIVX/const.NDIVY
	kxs=const.kx0
	kys=const.ky0
	phaseX=np.empty(const.noutX, dtype=complex)
	phaseY=np.empty(const.noutY, dtype=complex)
	fampxx = np.empty((const.noutX, const.noutY), dtype=complex)
	for ip in range(const.noutX):
		kxp=2.*np.pi*(ip-const.lpmaxX)/const.dx
		phaseX[ip]=np.exp(-1j*(kxs*kxp+kxp*kxp/2.)/const.k*const.z0)
	for jp in range(const.noutY):
		kyp=2.*np.pi*(jp-const.lpmaxY)/const.dy
		phaseY[jp]=np.exp(-1j*(kys*kyp+kyp*kyp/2.)/const.k*const.z0)
	for ip in range(const.noutX):
		for jp in range(const.noutY):
			fampxx[ip,jp]=fmask[ip,jp]*phaseX[ip]*phaseY[jp]/phase0

#	for ip in range(const.noutX):
#		for jp in range(const.noutY):
#			print(fampxx[ip,jp])

	nout=64 

	dod = descriptors.DiffractionOrderDescriptor(6.0)
	doc = diffraction_order.DiffractionOrderCoordinate(
		dod.max_diffraction_order_x,
		dod.max_diffraction_order_y,
		diffraction_order.rounded_diamond,
		)
	Nrange=doc.num_valid_diffraction_orders
	linput, minput, xinput, n_pupil_points=pupil.find_valid_pupil_points(Nrange,0,0)

# Mulipy SOCS kernels and a0,ax,ay in momentum space. The size is small +-32.
# Fourier interpolation will be used afterward to expand the image.
#	nsocs=50 # number of SOCS kernels for a0
	nsocs=100 # number of SOCS kernels for a0
	nsocsxy=20 # number of SOCS kernels for ax and ay
#	nsocsxy=-20 # number of SOCS kernels for ax and ay

	a0xx = np.empty((const.noutX, const.noutY), dtype=complex)
	axxx = np.empty((const.noutX, const.noutY), dtype=complex)
	ayxx = np.empty((const.noutX, const.noutY), dtype=complex)
	nxy=0
	for n in range(ncut):
		ip = linput[n] +const.lpmaxX
		jp = minput[n] +const.lpmaxY
		a0xx[ip, jp]=a0[n]
		if xinput[n] >=8:
			axxx[ip, jp]=ax[nxy]	
			ayxx[ip, jp]=ay[nxy]
			nxy=nxy+1	
		else:
			axxx[ip, jp]=0	
			ayxx[ip, jp]=0
#		print(a0xx[ip, jp],axxx[ip, jp],ayxx[ip, jp])

	Axs0 = np.empty(ncut, dtype=complex)	
	Axsx = np.empty(ncut, dtype=complex)
	Axsy = np.empty(ncut, dtype=complex)
	Axsxy = np.empty(ncut, dtype=complex)
	for n in range(ncut):
		kxplus = const.kx0+2*const.pi*linput[n]/const.dx/2
		kyplus = const.ky0+2*const.pi*minput[n]/const.dy/2
		kzplus = -np.sqrt(const.k*const.k-kxplus*kxplus-kyplus*kyplus)
		ip=linput[n]+const.lpmaxX
		jp=minput[n]+const.lpmaxY
		lp = ip-const.lpmaxX
		mp = jp-const.lpmaxY
		Axs0[n]= fampxx[ip,jp]+a0xx[ip,jp]
		Axsx[n]= axxx[ip,jp]*const.dx/2/np.pi
		Axsy[n]= ayxx[ip,jp]*const.dy/2/np.pi
		Axsxy[n]= axxx[ip,jp]*lp/2+ayxx[ip,jp]*mp/2

	intsmall=np.zeros((nout,nout), dtype=float)
	for m in range(nsocs):
		fns0=np.zeros((nout,nout), dtype=complex)
		for n in range(ncut):
			fs0=Axs0[n]*phiXs0[m,n]
			ix=linput[n]
			iy=minput[n]
			px=(ix+nout)%nout
			py=(iy+nout)%nout
			fns0[px,py]=fs0
		fns0=np.fft.ifft2(fns0)*nout*nout
		intsmall+=alphaXs0[m]*np.abs(fns0)**2
		if(m<nsocsxy):
			fns0x=np.zeros((nout,nout), dtype=complex)
			fns0y=np.zeros((nout,nout), dtype=complex)
			fnsx=np.zeros((nout,nout), dtype=complex)
			fnsy=np.zeros((nout,nout), dtype=complex)
			fnsxy=np.zeros((nout,nout), dtype=complex)
			for n in range(ncut):
				fs0x=Axs0[n]*phiXsx[m,n]
				fs0y=Axs0[n]*phiXsy[m,n]
				fsx=Axsx[n]*phiXsx[m,n]
				fsy=Axsy[n]*phiXsy[m,n]
				fsxy=Axsxy[n]*phiXs0[m,n]
				ix=linput[n]
				iy=minput[n]
				px=(ix+nout)%nout
				py=(iy+nout)%nout
				fns0x[px,py]=fs0x
				fns0y[px,py]=fs0y
				fnsx[px,py]=fsx
				fnsy[px,py]=fsy
				fnsxy[px,py]=fsxy
			fns0x=np.fft.ifft2(fns0x)*nout*nout
			fns0y=np.fft.ifft2(fns0y)*nout*nout
			fnsx=np.fft.ifft2(fnsx)*nout*nout
			fnsy=np.fft.ifft2(fnsy)*nout*nout
			fnsxy=np.fft.ifft2(fnsxy)*nout*nout
			intsmall+=(
			2*alphaXsx[m]*(fns0x.real*fnsx.real+fns0x.imag*fnsx.imag)
			+2*alphaXsy[m]*(fns0y.real*fnsy.real+fns0y.imag*fnsy.imag)
			+2*alphaXs0[m]*(fns0.real*fnsxy.real+fns0.imag*fnsxy.imag)
				)
	intsmall=np.fft.fft2(intsmall)/nout/nout
	XDIV=const.XDIV
	YDIV=const.YDIV
	intensity=np.zeros((XDIV,YDIV), dtype=complex)
	for i in range(-int(nout/2),int(nout/2)):
		for j in range(-int(nout/2),int(nout/2)):
			intensity[i,j]=intsmall[i,j]
#	cintensity=cp.asarray(intensity)
#	cintensity=cp.fft.ifft2(cintensity)
#	intensity=cintensity.get()*XDIV*YDIV
	intensity=np.fft.ifft2(intensity)*XDIV*YDIV
	return intensity

def socs(z,maskname,a0,ax,ay):
	directory="reflectivity"
	if not os.path.exists(directory):
		os.makedirs(directory)
		reflectivity=reflect_amplitude()
		with open(directory+"/reflectivity.txt","w") as f:
			f.write(str(reflectivity))
	with open(directory+"/reflectivity.txt","r") as f:
		data=eval(f.read())
		ampvc=data[0]
		ampab=data[1]
		phase0=data[2]
	directory="eigenfunction"
	if not os.path.exists(directory):
		os.makedirs(directory)
		eigenfunctions=eigenfunction(z)
		with open(directory+"/eigenfunctions.pkl","wb") as f:
			pickle.dump(eigenfunctions,f)		
	with open(directory+"/eigenfunctions.pkl","rb") as f:
		eigenfunctions=pickle.load(f)
		alphaXs0=eigenfunctions[0]
		phiXs0=eigenfunctions[1]
		alphaXsx=eigenfunctions[2]
		phiXsx=eigenfunctions[3]
		alphaXsy=eigenfunctions[4]
		phiXsy=eigenfunctions[5]

	with open(maskname, "rb") as f:
		bits=bitarray.bitarray()
		bits.fromfile(f)
	mask2d=bits.tolist()
#	mask2d=read_bits(maskname)
#	pattern=np.array(mask2d)*(ampab-ampvc)+ampvc*np.ones(const.NDIVX*const.NDIVY)
	pattern=np.array(mask2d)*(ampvc-ampab)+ampab*np.ones(const.NDIVX*const.NDIVY)
	pattern.resize(const.NDIVX,const.NDIVY)
	pattern=pattern.astype(np.complex64)
	cpattern=cp.asarray(pattern)

	start = time.perf_counter()
	intensity=intsocs(cpattern,a0,ax,ay,phase0,
		alphaXs0,phiXs0,alphaXsx,phiXsx,alphaXsy,phiXsy)
	end = time.perf_counter()
#	print(f"SOCS: {end-start: 5f}s")
	return intensity

if __name__ == "__main__":
	m3dpara=[]
	fpredict=open("inputxx.csv")
	predictreader=csv.reader(fpredict)
	for row in predictreader:
		amp=np.array(row)
		clean_amp=amp[amp !=""].astype(float)
		m3dpara.append(clean_amp)
	fpredict.close()
	a0re=m3dpara[0]
	a0im=m3dpara[1]
	axre=m3dpara[2]
	axim=m3dpara[3]
	ayre=m3dpara[4]
	ayim=m3dpara[5]
	a0=a0re+1j*a0im
	ax=axre+1j*axim
	ay=ayre+1j*ayim
	z=0 # defocus
	intensity=socs(z,"mask.bin",a0,ax,ay)

	XDIV=const.XDIV
	YDIV=const.YDIV
	with open("intsocs.csv","w") as f:
		for j in range(XDIV):
			for i in range(XDIV):
				value = np.real(intensity[i][XDIV-1-j])
				f.write(f"{value},")
			f.write("\n")

