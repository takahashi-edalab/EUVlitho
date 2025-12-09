import numpy as np
import cupy as cp
from elitho import const,pupil, electro_field, descriptors, diffraction_order

def eigenfunction(z):
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

	dod = descriptors.DiffractionOrderDescriptor(6.0)
	doc = diffraction_order.DiffractionOrderCoordinate(
		dod.max_diffraction_order_x,
		dod.max_diffraction_order_y,
		diffraction_order.rounded_diamond,
		)
	Nrange=doc.num_valid_diffraction_orders
	linput, minput, xinput, n_pupil_points=pupil.find_valid_pupil_points(Nrange,0,0)
	SDIV=len(dsx)
	ncut=len(linput)
	pmax=(const.k*const.NA)**2
	sx0=const.kx0
	sy0=const.ky0
	phaseX=np.empty((ncut, SDIV), dtype=complex)
	phase0X=np.empty((SDIV, ncut), dtype=complex)
	phasexX=np.empty((SDIV, ncut), dtype=complex)
	phaseyX=np.empty((SDIV, ncut), dtype=complex)
	phaseY=np.empty((ncut, SDIV), dtype=complex)
	phase0Y=np.empty((SDIV, ncut), dtype=complex)
	phasexY=np.empty((SDIV, ncut), dtype=complex)
	phaseyY=np.empty((SDIV, ncut), dtype=complex)
	phaseZ=np.empty((ncut, SDIV), dtype=complex)
	phase0Z=np.empty((SDIV, ncut), dtype=complex)
	phasexZ=np.empty((SDIV, ncut), dtype=complex)
	phaseyZ=np.empty((SDIV, ncut), dtype=complex)

	for i in range(ncut):
		kx=2*const.pi/const.dx*linput[i]
		ky=2*const.pi/const.dy*minput[i]
		for iso in range(SDIV):
			sx=dsx[iso]
			sy=dsy[iso]
			ksx=kx+sx
			ksy=ky+sy
			R=electro_field.polarization_rotation(const.k,const.MX,const.MY,ksx,ksy,sx0,sy0)
			phase=(np.exp(1j*((ksx+sx0)*(ksx+sx0)
			+(ksy+sy0)*(ksy+sy0))/2./const.k*const.z0
			+1j*(const.MX*const.MX*ksx*ksx
			+const.MY*const.MY*ksy*ksy)/2./const.k*z)
			*int((const.MX*const.MX*ksx*ksx
			+const.MY*const.MY*ksy*ksy)<=pmax))
			phaseX[i,iso]=phase*R[0,0]
			phaseY[i,iso]=phase*R[1,0]
			phaseZ[i,iso]=phase*R[2,0]
			tmp=const.k*const.k/(const.k*const.k-(sx0+sx)*(sx0+sx))
			phase0X[iso,i]=np.conj(phaseX[i,iso])*tmp
			phasexX[iso,i]=sx*np.conj(phaseX[i,iso])*tmp
			phaseyX[iso,i]=sy*np.conj(phaseX[i,iso])*tmp
			phase0Y[iso,i]=np.conj(phaseY[i,iso])*tmp
			phasexY[iso,i]=sx*np.conj(phaseY[i,iso])*tmp
			phaseyY[iso,i]=sy*np.conj(phaseY[i,iso])*tmp
			phase0Z[iso,i]=np.conj(phaseZ[i,iso])*tmp
			phasexZ[iso,i]=sx*np.conj(phaseZ[i,iso])*tmp
			phaseyZ[iso,i]=sy*np.conj(phaseZ[i,iso])*tmp
	cphaseX=cp.asarray(phaseX)
	cphase0X=cp.asarray(phase0X)
	cphasexX=cp.asarray(phasexX)
	cphaseyX=cp.asarray(phaseyX)
	cphaseY=cp.asarray(phaseY)
	cphase0Y=cp.asarray(phase0Y)
	cphasexY=cp.asarray(phasexY)
	cphaseyY=cp.asarray(phaseyY)
	cphaseZ=cp.asarray(phaseZ)
	cphase0Z=cp.asarray(phase0Z)
	cphasexZ=cp.asarray(phasexZ)
	cphaseyZ=cp.asarray(phaseyZ)
	TCCXS0=(cp.matmul(cphaseX,cphase0X)+cp.matmul(cphaseY,cphase0Y)
	+cp.matmul(cphaseZ,cphase0Z))/SDIV
	TCCXSX=(cp.matmul(cphaseX,cphasexX)+cp.matmul(cphaseY,cphasexY)
	+cp.matmul(cphaseZ,cphasexZ))/SDIV
	TCCXSY=(cp.matmul(cphaseX,cphaseyX)+cp.matmul(cphaseY,cphaseyY)
	+cp.matmul(cphaseZ,cphaseyZ))/SDIV
	calphaXs0, cphipXs0 = cp.linalg.eigh(TCCXS0)
	calphaXsx, cphipXsx = cp.linalg.eigh(TCCXSX)
	calphaXsy, cphipXsy = cp.linalg.eigh(TCCXSY)
	alphaXs0=calphaXs0.get()
	alphaXsx=calphaXsx.get()
	alphaXsy=calphaXsy.get()
	phipXs0=cphipXs0.get()
	phipXsx=cphipXsx.get()
	phipXsy=cphipXsy.get()

	phiXs0 = np.empty((ncut, ncut), dtype=complex)
	phiXsx = np.empty((ncut, ncut), dtype=complex)
	phiXsy = np.empty((ncut, ncut), dtype=complex)
	xs0=[]
	xsx=[]
	xsy=[]
	for i in range(ncut):
		xs0.append(i)
		xsx.append(i)
		xsy.append(i)
	for i in range(ncut-1):
		for j in range(ncut-1,i,-1):
			if np.abs(alphaXs0[j]) >np.abs(alphaXs0[j-1]):
				alp=alphaXs0[j]
				alphaXs0[j]=alphaXs0[j-1]
				alphaXs0[j-1]=alp
				tmp=xs0[j]
				xs0[j]=xs0[j-1]
				xs0[j-1]=tmp
			if np.abs(alphaXsx[j]) >np.abs(alphaXsx[j-1]):
				alp=alphaXsx[j]
				alphaXsx[j]=alphaXsx[j-1]
				alphaXsx[j-1]=alp
				tmp=xsx[j]
				xsx[j]=xsx[j-1]
				xsx[j-1]=tmp
			if np.abs(alphaXsy[j])>np.abs(alphaXsy[j-1]):
				alp=alphaXsy[j]
				alphaXsy[j]=alphaXsy[j-1]
				alphaXsy[j-1]=alp
				tmp=xsy[j]
				xsy[j]=xsy[j-1]
				xsy[j-1]=tmp
	for i in range(ncut):
		for j in range(ncut):
			phiXs0[i,j]=phipXs0[j,xs0[i]]
			phiXsx[i,j]=phipXsx[j,xsx[i]]
			phiXsy[i,j]=phipXsy[j,xsy[i]]

	return alphaXs0, phiXs0, alphaXsx, phiXsx, alphaXsy, phiXsy

"""
	for i in range(ncut):
		kx=2*const.pi/const.dx*linput[i]
		ky=2*const.pi/const.dy*minput[i]
		for j in range(ncut):
			kxp=2*const.pi/const.dx*linput[j]
			kyp=2*const.pi/const.dy*minput[j]
			sumxs0=0.
			sumxsx=0.
			sumxsy=0.
			for isource in range(SDIV):
				sx=dsx[isource]
				sy=dsy[isource]
				ksx=kx+sx
				ksy=ky+sy
				ksxp=kxp+sx
				ksyp=kyp+sy
				if (((const.MX*const.MX*ksx*ksx
				+const.MY*const.MY*ksy*ksy)<=pmax)
				&((const.MX*const.MX*ksxp*ksxp
				+const.MY*const.MY*ksyp*ksyp)<=pmax)):
					phase=np.exp(1j*((ksx+sx0)*(ksx+sx0)
			+(ksy+sy0)*(ksy+sy0))/2./const.k*const.z0
			+1j*(const.MX*const.MX*ksx*ksx
			+const.MY*const.MY*ksy*ksy)/2./const.k*z)
					phasep=np.exp(1j*((ksxp+sx0)*(ksxp+sx0)
			+(ksyp+sy0)*(ksyp+sy0))/2./const.k*const.z0
			+1j*(const.MX*const.MX*ksxp*ksxp
			+const.MY*const.MY*ksyp*ksyp)/2./const.k*z)
					sumxs0+=phase*(np.conj(phasep)
					/(const.k*const.k-(sx0+sx)*(sx0+sx)))
					sumxsx+=sx*phase*(np.conj(phasep)
					/(const.k*const.k-(sx0+sx)*(sx0+sx)))
					sumxsy+=sy*phase*(np.conj(phasep)
					/(const.k*const.k-(sx0+sx)*(sx0+sx)))
			TCCXS0[i,j] =sumxs0/SDIV
			TCCXSX[i,j] =sumxsx/SDIV
			TCCXSY[i,j] =sumxsy/SDIV
"""
