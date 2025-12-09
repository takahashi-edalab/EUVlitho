import numpy as np
import cupy as cp
import csv
import os
import time
import bitarray
from elitho import const,pupil, descriptors, diffraction_order, electro_field
from reflectivity import reflect_amplitude

def int_abbe(z,Ex,Ey,Ez,ncut,SDIV,dsx,dsy,linput,minput):
	k=const.k
	sx0=const.kx0
	sy0=const.ky0
	XDIV=const.XDIV
	YDIV=const.YDIV
	MX=const.MX
	MY=const.MY
	isum = np.zeros((XDIV, YDIV))
	for is_ in range(SDIV):
		h_fnx = np.zeros((XDIV, YDIV), dtype=np.complex128)
		h_fny = np.zeros((XDIV, YDIV), dtype=np.complex128)
		h_fnz = np.zeros((XDIV, YDIV), dtype=np.complex128)
		for n in range(ncut):
			kxn = dsx[is_] + 2.0 * np.pi * linput[n] / const.dx
			kyn = dsy[is_] + 2.0 * np.pi * minput[n] / const.dy
			if (MX*MX*kxn*kxn + MY*MY*kyn*kyn) <= (const.NA*k)**2:
				phase = np.exp(1j*((kxn+sx0)**2 + (kyn+sy0)**2)/(2*k)*const.z0
				 +1j*(MX*MX*kxn*kxn + MY*MY*kyn*kyn)/(2*k)*z)
				fx = Ex[is_,n] * phase
				fy = Ey[is_,n] * phase
				fz = Ez[is_,n] * phase
				ix = linput[n]
				iy = minput[n]
				px = (ix + XDIV) % XDIV
				py = (iy + YDIV) % YDIV
				h_fnx[px, py] = fx
				h_fny[px, py] = fy
				h_fnz[px, py] = fz
		h_fnx = np.fft.ifft2(h_fnx)*XDIV*YDIV
		h_fny = np.fft.ifft2(h_fny)*XDIV*YDIV
		h_fnz = np.fft.ifft2(h_fnz)*XDIV*YDIV
		isum += (np.abs(h_fnx)**2+np.abs(h_fny)**2+np.abs(h_fnz)**2)/SDIV
	return isum

def int_linear(z,cpattern,a0,ax,ay,phase0):
	ncut=a0.shape[0]  # number of (l,m) pairs for a0
	ncuts=ax.shape[0]  # number of (l,m) pairs for ax,ay
	mesh = 0.2 # source mesh (degree)
	kmesh=const.k*mesh*(const.pi/180.)
	skangx=const.k*const.NA/const.MX*const.sigma1
	skangy=const.k*const.NA/const.MY*const.sigma1
	l0max=int(skangx/kmesh)+1
	m0max=int(skangy/kmesh)+1
	dsx=[]
	dsy=[]
	for l in range(-l0max, l0max + 1):
                for m in range(-m0max, m0max + 1):
                    skx = l * kmesh
                    sky = m * kmesh
                    skxo = skx * const.MX
                    skyo = sky * const.MY
                    condition = False
                    if const.illumination_type == const.IlluminationType.CIRCULAR:
                        condition = (skxo**2 + skyo**2) <= (
                            const.k * const.NA * const.sigma1
                        ) ** 2
                    elif const.illumination_type == const.IlluminationType.ANNULAR:
                        r = np.sqrt(skxo**2 + skyo**2)
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        )
                    elif const.illumination_type == const.IlluminationType.DIPOLE_X:
                        r = np.sqrt(skxo**2 + skyo**2)
                        angle_condition = abs(skyo) <= abs(skxo) * np.tan(
                            const.pi * const.openangle / 180.0 / 2.0
                        )
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        ) and angle_condition
                    elif const.illumination_type == const.IlluminationType.DIPOLE_Y:
                        r = np.sqrt(skxo**2 + skyo**2)
                        angle_condition = abs(skxo) <= abs(skyo) * np.tan(
                            const.pi * const.openangle / 180.0 / 2.0
                        )
                        condition = (
                            const.k * const.NA * const.sigma2
                            <= r
                            <= const.k * const.NA * const.sigma1
                        ) and angle_condition
                    if condition:
                        dsx.append(skx)
                        dsy.append(sky)
	SDIV=len(dsx)

	dod = descriptors.DiffractionOrderDescriptor(6.0)
	doc = diffraction_order.DiffractionOrderCoordinate(
		dod.max_diffraction_order_x,
		dod.max_diffraction_order_y,
		diffraction_order.rounded_diamond,
		)
	Nrange=doc.num_valid_diffraction_orders
	linput, minput, xinput, n_pupil_points=pupil.find_valid_pupil_points(Nrange,0,0)

	a0xx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
	axxx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
	ayxx = np.zeros((const.noutX, const.noutY), dtype=np.complex128)
	nxy=0
	for n in range(ncut):
		ip = linput[n] + const.lpmaxX
		jp = minput[n] + const.lpmaxY
		a0xx[ip, jp] = a0[n]
		if xinput[n] >= 8:
			axxx[ip, jp] = ax[nxy]
			ayxx[ip, jp] = ay[nxy]
			nxy+=1
	cfpattern=cp.fft.fft2(cpattern)
	fpattern=cfpattern.get()

	fmask = np.empty((const.noutX, const.noutY), dtype=np.complex128)
	for i in range(const.noutX):
		l=i-const.lpmaxX
		for j in range(const.noutY):
			m=j-const.lpmaxY
			fmask[i,j]=fpattern[l,m]/const.NDIVX/const.NDIVY

	sx0=const.kx0
	sy0=const.ky0
	phaseX=np.empty(const.noutX, dtype=np.complex128)
	phaseY=np.empty(const.noutY, dtype=np.complex128)
	fampxx = np.empty((const.noutX, const.noutY), dtype=np.complex128)
	for ip in range(const.noutX):
		kxp=2.*np.pi*(ip-const.lpmaxX)/const.dx
		phaseX[ip]=np.exp(-1j*(sx0*kxp+kxp*kxp/2.)/const.k*const.z0)
	for jp in range(const.noutY):
		kyp=2.*np.pi*(jp-const.lpmaxY)/const.dy
		phaseY[jp]=np.exp(-1j*(sy0*kyp+kyp*kyp/2.)/const.k*const.z0)
	for ip in range(const.noutX):
		for jp in range(const.noutY):
			fampxx[ip,jp]=fmask[ip,jp]*phaseX[ip]*phaseY[jp]/phase0

	k=const.k
	dx=const.dx
	dy=const.dy
	Efx = np.zeros((SDIV, ncut), dtype=np.complex128)
	Efy = np.zeros((SDIV, ncut), dtype=np.complex128)
	Efz = np.zeros((SDIV, ncut), dtype=np.complex128)
	Ex = np.zeros((SDIV, ncut), dtype=np.complex128)
	Ey = np.zeros((SDIV, ncut), dtype=np.complex128)
	Ez = np.zeros((SDIV, ncut), dtype=np.complex128)
	for is_ in range(SDIV):
		kx = sx0 + dsx[is_]
		ky = sy0+ dsy[is_]
		ls = dsx[is_] / (2.0 * np.pi / dx)
		ms = dsy[is_] / (2.0 * np.pi / dy)
		for i in range(ncut):
			kxplus = kx + 2 * np.pi * linput[i] / dx
			kyplus = ky + 2 * np.pi * minput[i] / dy
			ip = linput[i] + const.lpmaxX
			jp = minput[i] + const.lpmaxY
			lp = ip - const.lpmaxX
			mp = jp - const.lpmaxY
			Afx = fampxx[ip, jp] / np.sqrt(k * k - kx * kx)
			Ax = (fampxx[ip, jp] + a0xx[ip, jp] +
				axxx[ip, jp] * (ls + lp / 2.0) +
				ayxx[ip, jp] * (ms + mp / 2.0)) / np.sqrt(k * k - kx * kx)
			Ay = 0.0

			EAfx,EAfy,EAfz= electro_field.high_na_electro_field(0, 0, Afx, Ay, lp, mp, ls, ms)
#			EAfx,EAfy,EAfz= electro_field.standard_na_electro_field(kxplus, kyplus,Afx, Ay)

			Efx[is_,i] = EAfx
			Efy[is_,i] = EAfy
			Efz[is_,i] = EAfz

			EAx,EAy,EAz= electro_field.high_na_electro_field(0, 0, Ax, Ay, lp, mp, ls, ms)
#			EAx,EAy,EAz= electro_field.standard_na_electro_field(kxplus, kyplus,Ax, Ay)

			Ex[is_,i] = EAx
			Ey[is_,i] = EAy
			Ez[is_,i] = EAz

	intft=int_abbe(z,Efx,Efy,Efz,ncut,SDIV,dsx,dsy,linput,minput)
	intlinear=int_abbe(z,Ex,Ey,Ez,ncut,SDIV,dsx,dsy,linput,minput)
	return intlinear,intft

def linear(z,maskname,re0,im0,rex,imx,rey,imy):
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

	with open(maskname, "rb") as f:
		bits=bitarray.bitarray()
		bits.fromfile(f)
	mask2d=bits.tolist()
#	mask2d=read_bits(maskname)
#	pattern=np.array(mask2d)*(ampab-ampvc)+ampvc*np.ones(const.NDIVX*const.NDIVY)
	pattern=np.array(mask2d)*(ampvc-ampab)+ampab*np.ones(const.NDIVX*const.NDIVY)
	pattern.resize(const.NDIVX,const.NDIVY)
	pattern=pattern.astype(np.complex128)
	cpattern=cp.asarray(pattern)
	a0=re0+1j*im0
	ax=rex+1j*imx
	ay=rey+1j*imy
	start = time.perf_counter()
	intlinear, intft =int_linear(z,cpattern,a0,ax,ay,phase0)
	end = time.perf_counter()
#	print(f"linear: {end-start: 5f}s")
	return intlinear,intft

if __name__ == "__main__":
	z=0 # defocus
	m3dpara=[]
	fpredict=open("inputxx.csv")
	predictreader=csv.reader(fpredict)
	for row in predictreader:
		amp=np.array(row)
		clean_amp=amp[amp !=""].astype(float)
		m3dpara.append(clean_amp)
	fpredict.close()
	re0=m3dpara[0]
	im0=m3dpara[1]
	rex=m3dpara[2]
	imx=m3dpara[3]
	rey=m3dpara[4]
	imy=m3dpara[5]
	intlinear,intft=linear(z,"mask.bin",re0,im0,rex,imx,rey,imy)

	XDIV=const.XDIV
	YDIV=const.YDIV
	with open("intlinear.csv","w") as f:
		for j in range(XDIV):
			for i in range(XDIV):
				value = np.real(intlinear[i][XDIV-1-j])
				f.write(f"{value},")
			f.write("\n")
	with open("intft.csv","w") as f:
		for j in range(XDIV):
			for i in range(XDIV):
				value = np.real(intft[i][XDIV-1-j])
				f.write(f"{value},")
			f.write("\n")

