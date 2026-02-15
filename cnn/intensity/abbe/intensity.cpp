#include <iostream>
#include<fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define _USE_MATH_DEFINES
#define OPENBLAS_NUM_THREADS 96
#define OMP_NUM_THREADS 96
#include <omp.h>
#include "Eigen/Eigen"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "../../include/header.h"

vector<bool> decompressBits(vector<uint8_t>& bytes, size_t originalBitCount) {
    vector<bool> bits;
    for (uint8_t byte : bytes) {
        for (int i = 7; i >= 0; --i) {
            bits.push_back((byte >> i) & 1);
            if (bits.size() == originalBitCount) break; 
        }
    }
    return bits;
}

void rotate(double k,int MX,int MY,double px,double py,double sx0,double sy0,Eigen::MatrixXd& R);
void source(double NA,int type,double sigma1,double sigma2,double openangle,double k,double dx,double dy,
int&ndivs, vector<vector<vector<int>>>& l0s,vector<vector<vector<int>>>& m0s,vector<vector<int>>& SDIV,int& MX,int& MY);

int main (int argc,char* argv[])
{
 std::chrono::system_clock::time_point  start, now;
 double elapsed;
 start = std::chrono::system_clock::now(); 

 char com;
 ofstream ofsint;

 int MX=4; //Mask magnification in X
 int MY=4;
// int MY=8; //Mask magnification in Y
// int NDIVX=1024; //Mask size in X
 int NDIVX=2048;
 int NDIVY=NDIVX;
// int NDIVY =2*NDIVX; //Mask size in Y
 double dx =NDIVX;
 double dy = NDIVY;
 int XDIV=NDIVX/MX;
 double pi=atan(1.)*4.;
 complex<double> zi (0., 1.);
 double lambda,theta0,azimuth;
 lambda = 13.5;
 double k = 2.*pi/lambda;
 theta0 = -6.;
// theta0 = -5.3; //Incident angle
 azimuth =0.;  //Azimuthal angle
// azimuth =90.; 
 double phi0;
 phi0 = 90. - azimuth;
 double NA,sigma1,sigma2,openangle;
 NA = 0.33;
// NA=0.55;
 int type=2;
// int type=3; //Illumination type: 0 conventional, 1 annular, 2 X dipole, 3 Y dipole
 sigma1 = 0.9; //Outer sigma
 sigma2=0.55; //Inner sigma
// sigma1 = 0.85;
// sigma2=0.6;
 openangle = 90.; //Open angle of dipole illumination
   int zmin,zmax,dz; //defocus
   zmin = 0;
   zmax = 0;
   dz = 10;
  double dabst=60.; //absorber thickness
  double z0=dabst+42.;  //mask defocus (reflecton plane inside ML)
   complex<double> nta(0.9567,0.0343); //Refractive index of Ta absorber
   vector< complex<double>> eabs(100);
   vector<double> dabs(100); 
   int NML=40; //number of Mo/Si pairs in ML
   int NABS=1; //number of absorber layers
   eabs[0]=nta*nta;
   dabs[0]= dabst;

// int ndivs=4;
  double sigmadiv=0.5; 
 int ndivs=max(1,int(180./pi*lambda/dx/sigmadiv));
 vector<vector<vector<int>>> l0s(ndivs,vector<vector<int>>(ndivs,vector<int>()));
 vector<vector<vector<int>>> m0s(ndivs,vector<vector<int>>(ndivs,vector<int>()));
 vector<vector<int>> SDIV(ndivs,vector<int>(ndivs));
 source(NA,type,sigma1,sigma2,openangle,k,dx,dy,ndivs,l0s,m0s,SDIV,MX,MY);
 int SDIVMAX=0;
 int SDIVSUM=0;
 for(int nsx=0;nsx<ndivs;nsx++)
 for(int nsy=0;nsy<ndivs;nsy++)
 {
  SDIVMAX=max(SDIVMAX,SDIV[nsx][nsy]);
  SDIVSUM=SDIVSUM+SDIV[nsx][nsy];
//  cout<<SDIV[nsx][nsy]<<endl;
 }

 double delta=1.;
// double delta=0.5;
 int FDIVX=dx/delta+0.000001; 
 int FDIVY=dy/delta+0.000001; 
// int FDIVY=dx/delta+0.000001; 
 int NDIVSQ=NDIVX*NDIVY;
 int* mask2d=new int[NDIVSQ];

 ifstream imask("./mask.bin",ios::binary);
 vector<char> buffer(NDIVSQ/8);
 imask.read(buffer.data(),NDIVSQ/8);

 Eigen::MatrixXcd pattern(FDIVX,FDIVY);
 int lsmaxX=NA*dx/double(MX)/lambda+1;
 int lsmaxY=NA*dy/double(MY)/lambda+1;
 int lpmaxX=NA*dx/double(MX)*2/lambda+0.0001;
 int lpmaxY=NA*dy/double(MY)*2/lambda+0.0001;
 int nsourceX=2*lsmaxX+1;
 int nsourceY=2*lsmaxY+1;
 int noutX=2*lpmaxX+1;
 int noutY=2*lpmaxY+1;
 int nsourceXL=2*lsmaxX+10;
 int nsourceYL=2*lsmaxY+10;
 int noutXL=2*lpmaxX+10;
 int noutYL=2*lpmaxY+10;

 int FDIVX1 = FDIVX + 1;
 int FDIVY1 = FDIVY + 1;
 vector< complex<double> > cexpX(FDIVX1),cexpY(FDIVY1);
 exponential(cexpX, pi, FDIVX);
 exponential(cexpY, pi, FDIVY);

 double cutx,cuty;
 cutx = NA/MX*6.;
 cuty = NA/MY*6.;;
 int LMAX = cutx*dx / lambda;
 int Lrange = 2 * LMAX + 1;
 int Lrange2 = 4 * LMAX + 1;
 int MMAX = cuty*dy / lambda;
 int Mrange = 2 * MMAX + 1;
 int Mrange2 = 4 * MMAX + 1;

 int Nrange = 0;
 vector<int> lindex, mindex;
 for (int i = 0; i < Lrange; i++)
 {
  int ii = i - LMAX;
  for (int j = 0; j < Mrange; j++)
  {
   int jj = j - MMAX;
   if ((abs(ii) / double(LMAX + 0.01) + 1.)*(abs(jj) / double(MMAX + 0.01) + 1.) <= 2.)
//   if (pow(ii/ double(LMAX + 0.01), 2.)+pow(jj/ double(MMAX + 0.01), 2.) <= 1.)
   {
    lindex.push_back(ii);
    mindex.push_back(jj);
    Nrange++;
   }
  }
 }

 int ninput=0;
 vector<int> linput(Nrange),minput(Nrange);
 for(int ip=0;ip<noutX;ip++) 
 {
  for(int jp=0;jp<noutY;jp++) 
  {
   int snum=0;
   for(int is=0;is<nsourceX;is++)
   {
     for(int js=0;js<nsourceY;js++)
     {
      if(((pow((is-lsmaxX)*MX/dx,2)
           +pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2))
          &&((pow((ip-lpmaxX+is-lsmaxX)*MX/dx,2)
           +pow((jp-lpmaxY+js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)))
       {

        snum+=1;
       }
     }
    }
   if(snum>0)
   {
    linput[ninput]=ip-lpmaxX;
    minput[ninput]=jp-lpmaxY;
    ninput++;
   }
  }
 }
 int ncut=ninput;

 vector<uint8_t> bytes(NDIVSQ/8);
 vector<bool> bits(NDIVSQ);
 for (int i=0;i<NDIVSQ/8;i++)
 {
  bytes[i]=static_cast<uint8_t>(buffer[i]);
 } 
 bits= decompressBits(bytes, NDIVSQ);
 for(int i=0;i<NDIVSQ;i++)
 {
  mask2d[i]=1-static_cast<int>(bits[i]);
 }

 cufftDoubleComplex *h_fnx, *d_fnx,*h_fny, *d_fny,*h_fnz, *d_fnz;
 cudaMallocHost((void **)&h_fnx, sizeof(cufftDoubleComplex)*XDIV*XDIV);
 cudaMallocHost((void **)&h_fny, sizeof(cufftDoubleComplex)*XDIV*XDIV);
 cudaMallocHost((void **)&h_fnz, sizeof(cufftDoubleComplex)*XDIV*XDIV);
 cudaMalloc((void **)&d_fnx, sizeof(cufftDoubleComplex)*XDIV*XDIV);
 cudaMalloc((void **)&d_fny, sizeof(cufftDoubleComplex)*XDIV*XDIV);
 cudaMalloc((void **)&d_fnz, sizeof(cufftDoubleComplex)*XDIV*XDIV);
 cufftHandle plan;
 cufftPlan2d(&plan,XDIV,XDIV, CUFFT_Z2Z);

for (int z = zmin; z <= zmax; z = z + dz)
{
 vector<vector<vector<vector<vector<double>>>>> isum(ndivs,vector<vector<vector<vector<double>>>>
  (ndivs,vector<vector<vector<double>>>(XDIV,vector<vector<double>>(XDIV,vector<double>(SDIVMAX)))));

for (int ipl = 0; ipl<=0; ipl++)
{
 if(ipl==0)
  ofsint.open("emint.csv");
 else
  ofsint.open("emintY.csv");

for(int nsx=0;nsx<ndivs;nsx++)
for(int nsy=0;nsy<ndivs;nsy++)
{
 double kx0,ky0,sx0, sy0;
 kx0 = k*sin(pi / 180.*theta0)*cos(pi / 180.*phi0);
 ky0 = k*sin(pi / 180.*theta0)*sin(pi / 180.*phi0);
 sx0 = 2.*pi/dx*nsx/double(ndivs)+kx0;
 sy0 = 2.*pi/dy*nsy/double(ndivs)+ky0;

  vector<vector<Eigen::VectorXcd>> Ax(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrange)));
 if(ipl==0) 
  ampS('X',Ax, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrange, lindex, mindex, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, sx0, sy0, eabs, dabs, cexpX,cexpY);
 else 
  ampS('Y',Ax, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrange, lindex, mindex, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, sx0, sy0, eabs, dabs, cexpX,cexpY);

 vector<vector<Eigen::MatrixXcd>>  ampxx(nsourceXL, vector<Eigen::MatrixXcd>(nsourceYL));
 for(int is=0;is<nsourceXL;is++)
 for(int js=0;js<nsourceYL;js++)
  {
    ampxx[is][js].resize(noutXL,noutYL);
     for(int ip=0;ip<noutXL;ip++)
    for(int jp=0;jp<noutYL;jp++)
          ampxx[is][js](ip,jp)=-1000.;
 } 

  for(int is=0;is<nsourceXL;is++)
  {
   for(int js=0;js<nsourceYL;js++)
   {
    if((pow((is-lsmaxX)*MX/dx,2)+pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)*1.0)
    {
     for(int n=0;n<Nrange;n++)
     {
      int ip=lindex[n]-(is-lsmaxX)+lpmaxX;
      int jp=mindex[n]-(js-lsmaxY)+lpmaxY;
      if((0<=ip)&&(ip<noutXL)&&(0<=jp)&&(jp<noutYL))
//      if((pow(lindex[n]*MX/dx,2)+pow(mindex[n]*MY/dy,2))<=pow(NA/lambda,2)*1.2)
       ampxx[is][js](ip,jp)=Ax[is][js](n);
     }
    }
   }
  }

  vector<Eigen::VectorXcd> Ex0m(SDIV[nsx][nsy], Eigen::VectorXcd(ncut)), Ey0m(SDIV[nsx][nsy], Eigen::VectorXcd(ncut)),
   Ez0m(SDIV[nsx][nsy], Eigen::VectorXcd(ncut));
 for (int is = 0; is < SDIV[nsx][nsy]; is++)
 {
  double kx, ky;
  kx = sx0 + 2.*pi/dx*l0s[nsx][nsy][is];
  ky = sy0 + 2.*pi/dy*m0s[nsx][nsy][is];
  int ls=l0s[nsx][nsy][is]+lsmaxX;
  int ms=m0s[nsx][nsy][is]+lsmaxY;
  for (int i = 0; i < ncut; i++)
  {
   double kxplus, kyplus, kxy2;
   complex<double> klm;
   kxplus = kx + 2 * pi*linput[i] / dx;
   kyplus = ky + 2 * pi*minput[i] / dy;
   kxy2 = pow(kxplus, 2) + pow(kyplus, 2);
   klm = sqrt(k*k - kxy2);
   complex<double>  Ax, Ay;
   int ip,jp;
   ip=linput[i]+lpmaxX;
   jp=minput[i]+lpmaxY;
   if(ipl==0)
   {
    if(real(ampxx[ls][ms](ip,jp))<-100) cout<<ip-lpmaxX+ls-lsmaxX<<","<<jp-lpmaxY+ms-lsmaxY<<endl;
    Ax= ampxx[ls][ms](ip,jp)/sqrt(k*k-kx*kx);
    Ay=0.;
   }
   else if(ipl==1)
   {
    Ay= ampxx[ls][ms](ip,jp)/sqrt(k*k-ky*ky);
    Ax=0.;
   }

   complex<double>  EAx, EAy, EAz;
//   EAx=zi*k*Ax-zi/k*(pow(kxplus,2.)*Ax+kxplus*kyplus*Ay);
//   EAy=zi*k*Ay-zi/k*(kxplus*kyplus*Ax+pow(kyplus,2.)*Ay);
//   EAz=zi*klm/k*(kxplus*Ax+kyplus*Ay);
   Eigen::MatrixXd R(3,2);
 double kxn,kyn;
  kxn = 2.*pi/dx*nsx/double(ndivs)+2.*pi/dx*l0s[nsx][nsy][is]  + 2.*pi*linput[i] / dx ;
  kyn = 2.*pi/dy*nsy/double(ndivs)+2.*pi/dy*m0s[nsx][nsy][is]  + 2.*pi*minput[i] / dy ;
  if ((MX*MX*kxn*kxn+MY*MY*kyn*kyn) <= pow(NA*k,2))
  {
   rotate(k,MX,MY,kxn,kyn,kx0,ky0,R);
//  cout<<k<<","<<MX<<","<<MY<<","<<kxplus<<","<<kyplus<<","<<kx0<<","<<ky0<<endl;
//    cout<<R(0,0)<<","<<R(0,1)<<","<<R(1,0)<<","<<R(1,1)<<","<<R(2,0)<<","<<R(2,1)<<endl;
//    return 0;
   EAx=zi*k*(R(0,0)*Ax+R(0,1)*Ay);
   EAy=zi*k*(R(1,0)*Ax+R(1,1)*Ay);
   EAz=zi*k*(R(2,0)*Ax+R(2,1)*Ay);
   Ex0m[is](i) = EAx;
   Ey0m[is](i) = EAy;
   Ez0m[is](i) = EAz;
  }
 }
 }

 for (int is = 0; is < SDIV[nsx][nsy]; is++)
 {
 for(int i=0;i<XDIV*XDIV;i++)
 {
  h_fnx[i].x=0.;
  h_fnx[i].y=0.;
  h_fny[i].x=0.;
  h_fny[i].y=0.;
  h_fnz[i].x=0.;
  h_fnz[i].y=0.;
 }
 complex <double> fx,fy,fz;
 double kxn,kyn;
 for (int n = 0; n < ncut; n++)
 {
  kxn = 2.*pi/dx*nsx/double(ndivs)+2.*pi/dx*l0s[nsx][nsy][is]  + 2.*pi*linput[n] / dx ;
  kyn = 2.*pi/dy*nsy/double(ndivs)+2.*pi/dy*m0s[nsx][nsy][is]  + 2.*pi*minput[n] / dy ;
  if ((MX*MX*kxn*kxn+MY*MY*kyn*kyn) <= pow(NA*k,2))
  {
   complex<double> phase;
   phase=exp(zi*((kxn+kx0)*(kxn+kx0)+(kyn+ky0)*(kyn+ky0))/2./k*z0+zi*(MX*MX*kxn*kxn+MY*MY*kyn*kyn)/2./k*double(z));
   fx=Ex0m[is](n)*phase;
   fy=Ey0m[is](n)*phase;
   fz=Ez0m[is](n)*phase;
   int ix=linput[n];
   int iy=minput[n];
   int px=(ix+XDIV)%XDIV;
   int py=(iy+XDIV)%XDIV;
    h_fnx[px*XDIV+py].x=real(fx);
    h_fnx[px*XDIV+py].y=imag(fx);
    h_fny[px*XDIV+py].x=real(fy);
    h_fny[px*XDIV+py].y=imag(fy);
    h_fnz[px*XDIV+py].x=real(fz);
    h_fnz[px*XDIV+py].y=imag(fz);
  }
 }
  cudaMemcpy(d_fnx, h_fnx, sizeof(cufftDoubleComplex)*XDIV*XDIV, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, d_fnx, d_fnx, CUFFT_INVERSE);
  cudaMemcpy(h_fnx, d_fnx, sizeof(cufftDoubleComplex)*XDIV*XDIV, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_fny, h_fny, sizeof(cufftDoubleComplex)*XDIV*XDIV, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, d_fny, d_fny, CUFFT_INVERSE);
  cudaMemcpy(h_fny, d_fny, sizeof(cufftDoubleComplex)*XDIV*XDIV, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_fnz, h_fnz, sizeof(cufftDoubleComplex)*XDIV*XDIV, cudaMemcpyHostToDevice);
  cufftExecZ2Z(plan, d_fnz, d_fnz, CUFFT_INVERSE);
  cudaMemcpy(h_fnz, d_fnz, sizeof(cufftDoubleComplex)*XDIV*XDIV, cudaMemcpyDeviceToHost);

 for(int i=0;i<XDIV;i++)
 for(int j=0;j<XDIV;j++)
 {
   isum[nsx][nsy][i][j][is] = (h_fnx[i*XDIV+j].x*h_fnx[i*XDIV+j].x+h_fnx[i*XDIV+j].y*h_fnx[i*XDIV+j].y
        +h_fny[i*XDIV+j].x*h_fny[i*XDIV+j].x+h_fny[i*XDIV+j].y*h_fny[i*XDIV+j].y
        +h_fnz[i*XDIV+j].x*h_fnz[i*XDIV+j].x+h_fnz[i*XDIV+j].y*h_fnz[i*XDIV+j].y);
//    /double(XDIV)/double(XDIV)/double(XDIV)/double(XDIV);
  }
 }
 }

 for(int j=XDIV-1;j>=0;j--)
 {
  for(int i=0;i<XDIV;i++)
  {
   double sum(0.);
   for(int nsx=0;nsx<ndivs;nsx++)
   for(int nsy=0;nsy<ndivs;nsy++)
   {
//   #pragma omp parallel for reduction(+:sum)
    for (int is = 0; is < SDIV[nsx][nsy]; is++)
    {
    sum += isum[nsx][nsy][i][j][is];
    }
   }
   ofsint<<sum /SDIVSUM << ",";
  }
  ofsint << endl;
 }
  ofsint.close();
 }
}
 cudaFree(h_fnx);
 cudaFree(h_fny);
 cudaFree(h_fnz);
 cudaFree(d_fnx);
 cudaFree(d_fny);
 cudaFree(d_fnz);
 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
 cout<<"elapsed time (s) "<<elapsed/1000.<<endl;
 return 0;
}

void rotate(double k,int MX,int MY,double px,double py,double sx0,double sy0,Eigen::MatrixXd& R)
{
 Eigen::Vector3d s0,p,pp,ps0,es,esp,em,emp,ez,ezp;
 s0(0)=sx0;
 s0(1)=sy0;
 s0(2)=-sqrt(k*k-sx0*sx0-sy0*sy0);
 p(0)=px;
 p(1)=py;
 p(2)=0.;
 pp(0)=MX*px;
 pp(1)=MY*py;
 pp(2)=-sqrt(k*k-MX*px*MX*px-MY*py*MY*py);
 ez(0)=0.;
 ez(1)=0.;
 ez(2)=1.;
 ezp(0)=0.;
 ezp(1)=0.;
 ezp(2)=1.;
 double eps=0.00001;
 if((px*px+py*py)>eps)
 {
  es=p.cross(s0);
  es=es/es.norm();
  esp=pp.cross(ezp);
  esp=esp/esp.norm();
  ps0=p+s0;
  ps0=ps0/ps0.norm();
  em=es.cross(ps0);
  emp=esp.cross(pp)/k;
 }
 else
 {
  es=ez.cross(s0);
  es=es/es.norm();
  esp=-es;
  em=es.cross(s0)/k;
  emp=esp.cross(-ezp);
 }
 for(int i=0;i<=2;i++)
 for(int j=0;j<=1;j++)
   R(i,j)=sqrt(k/abs(pp(2)))*(esp(i)*es(j)+emp(i)*em(j));
}

void source(double NA,int type,double sigma1,double sigma2,double openangle,double k,double dx,double dy,
int& ndivs, vector<vector<vector<int>>>& l0s,vector<vector<vector<int>>>& m0s,vector<vector<int>>& SDIV,int& MX,int& MY)
{
 double pi=atan(1.)*4.;
 double dkxang=2.*pi/dx;
 double dkyang=2.*pi/dy;
 double skangx=k*NA/MX*sigma1;
 double skangy=k*NA/MY*sigma1;
 int l0max=skangx/dkxang+1;
 int m0max=skangy/dkyang+1;

 for(int nsx=0;nsx<ndivs;nsx++)
 for(int nsy=0;nsy<ndivs;nsy++)
 {
  SDIV[nsx][nsy]=0;
   for(int l=-l0max;l<=l0max;l++)
   for(int m=-m0max;m<=m0max;m++)
    {
      double skx=l*dkxang+2.*pi/dx*nsx/double(ndivs);
      double sky=m*dkyang+2.*pi/dy*nsy/double(ndivs);
      double skxo=skx*MX;
      double skyo=sky*MY;
      if(((type==0)&&(skxo*skxo+skyo*skyo)<=pow(k*NA*sigma1,2))
         ||((type==1)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2))
         ||((type==2)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2)
	      &&(abs(skyo)<=abs(skxo)*tan(pi*openangle/180./2.)))
         ||((type==3)&&(sqrt(skxo*skxo+skyo*skyo)<=k*NA*sigma1)&&(sqrt(skxo*skxo+skyo*skyo)>=k*NA*sigma2)
	      &&(abs(skxo)<=abs(skyo)*tan(pi*openangle/180./2.))))
        {
         l0s[nsx][nsy].push_back(l);
         m0s[nsx][nsy].push_back(m);
         SDIV[nsx][nsy]++;
        }
    }
  }
}


