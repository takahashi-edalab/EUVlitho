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
#include "magma_v2.h"
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

int main (int argc,char* argv[])
{
 std::chrono::system_clock::time_point  start, now;
 double elapsed;
 start = std::chrono::system_clock::now(); 
 magma_init();

 int ndata=1;
// srand(time(NULL));

 char com;
 double pi=atan(1.)*4.;
 complex<double> zi (0., 1.);
// ofstream ofsrange("nrange.csv");
 ofstream ofsxx("inputxx.csv");
// ofstream ofsyy("inputyy.csv");

 double lambda,theta0,azimuth;
 lambda = 13.5;
 double k = 2.*pi/lambda;
 double NA =0.33;
//double NA = 0.55;
 theta0 = -6.;
// theta0 = -5.3;
 azimuth =0.;  
//azimuth =90.; 
 double phi0;
 phi0 = 90. - azimuth;
 double kx0, ky0;
 kx0 = k*sin(pi / 180.*theta0)*cos(pi / 180.*phi0);
 ky0 = k*sin(pi / 180.*theta0)*sin(pi / 180.*phi0);
 int MX=4;
// int MY=8;
 int MY=4;
 int NDIVX =2048;
// int NDIVX =1024;
 int NDIVY =NDIVX;
// int NDIVY =2*NDIVX;
 double dx =NDIVX;
 double dy =NDIVY;
 double delta=1.;
 int FDIVX=dx/delta+0.000001; 
 int FDIVY=dy/delta+0.000001; 
 int FDIVX1 = FDIVX + 1;
 int FDIVY1 = FDIVY + 1;
 vector< complex<double> > cexpX(FDIVX1),cexpY(FDIVY1);
 exponential(cexpX, pi, FDIVX);
 exponential(cexpY, pi, FDIVY);
 int lsmaxX=NA/MX*dx/lambda+1;
 int lsmaxY=NA/MY*dy/lambda+1;
 int lpmaxX=NA*dx/double(MX)*2/lambda+0.0001;
 int lpmaxY=NA*dy/double(MY)*2/lambda+0.0001;
 int nsourceX=2*lsmaxX+1;
 int nsourceY=2*lsmaxY+1;
 int noutX=2*lpmaxX+1;
 int noutY=2*lpmaxY+1;
 int NDIVSQ=NDIVX*NDIVY;
 int* mask2d=new int[NDIVSQ];
 int* mask2d0=new int[NDIVSQ];

 ifstream imask("./mask.bin",ios::binary);
 vector<char> buffer(NDIVSQ/8*ndata);
 imask.read(buffer.data(),NDIVSQ/8*ndata);

 complex<double> nta(0.9567,0.0343);
 vector< complex<double>> eabs(100);
 vector<double> dabs(100); 
 double dabst=60.;
 double z0=dabst+42.; 
 int NML=40;
 int NABS=1;
 eabs[0]=nta*nta;
 dabs[0]= dabst;

 double cutx,cuty;
// cutx = NA/MX*6.;
// cuty = NA/MY*6.;
 cutx = NA/MX*1.5;
 cuty = NA/MY*1.5;
 int LMAX = cutx*dx / lambda;
 int Lrange = 2 * LMAX + 1;
 int Lrange2 = 4 * LMAX + 1;
 int MMAX = cuty*dy / lambda;
 int Mrange = 2 * MMAX + 1;
 int Mrange2 = 4 * MMAX + 1;

 int Nrangep = 0;
 vector<int> lindexp, mindexp;
 for (int i = 0; i < Lrange; i++)
 {
  int ii = i - LMAX;
  for (int j = 0; j < Mrange; j++)
  {
   int jj = j - MMAX;
//   if ((abs(ii) / double(LMAX + 0.01) + 1.)*(abs(jj) / double(MMAX + 0.01) + 1.) <= 2.)
   if (pow(ii/ double(LMAX + 0.01), 2.)+pow(jj/ double(MMAX + 0.01), 2.) <= 1.)
   {
    lindexp.push_back(ii);
    mindexp.push_back(jj);
    Nrangep++;
   }
  }
 }

 cutx = NA/MX*6.;
 cuty = NA/MY*6.;;
 LMAX = cutx*dx / lambda;
 Lrange = 2 * LMAX + 1;
 Lrange2 = 4 * LMAX + 1;
 MMAX = cuty*dy / lambda;
 Mrange = 2 * MMAX + 1;
 Mrange2 = 4 * MMAX + 1;

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
//    ofsrange<<ii<<","<<jj<<endl;
    Nrange++;
   }
  }
 }
// ofsrange.close();

 ofstream ofslm("inputlm.csv");
 int ninput=0;
 vector<int> linput(Nrange),minput(Nrange),xinput(Nrange);
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
    if(snum>=8)
//    if(snum>=12)
      xinput[ninput]=1;
    else
      xinput[ninput]=0;        
    ofslm<<linput[ninput]<<","<<minput[ninput]<<","<<xinput[ninput]<<endl;          
    ninput++;
   }
  }
 }
 ofslm.close();

for(int nsample=0;nsample<ndata;nsample++)
{
 vector<uint8_t> bytes(NDIVSQ/8);
 vector<bool> bits(NDIVSQ);
 for (int i=0;i<NDIVSQ/8;i++)
 {
  bytes[i]=static_cast<uint8_t>(buffer[i+NDIVSQ/8*nsample]);
 } 
 bits= decompressBits(bytes, NDIVSQ);
 for(int i=0;i<NDIVSQ;i++)
 {
  mask2d[i]=1-static_cast<int>(bits[i]);
 }

//for(int npol=0;npol<=1;npol++)
for(int npol=0;npol<=0;npol++)
{
 vector<vector<Eigen::VectorXcd>> Axvc(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrangep)));
 for(int i=0;i<NDIVX;i++)
  {
   for(int j=0;j<NDIVY;j++)
    mask2d0[NDIVY*i+j]=0;
  }
 if(npol==0)
  ampS('X',Axvc, NDIVX, NDIVY, mask2d0, LMAX, Lrange2, MMAX, Mrange2, Nrangep, lindexp, mindexp, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, kx0, ky0, eabs, dabs, cexpX,cexpY);
 else
  ampS('Y',Axvc, NDIVX, NDIVY, mask2d0, LMAX, Lrange2, MMAX, Mrange2, Nrangep, lindexp, mindexp, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, kx0, ky0, eabs, dabs, cexpX,cexpY);

 vector<vector<Eigen::VectorXcd>> Axab(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrangep)));
 for(int i=0;i<NDIVX;i++)
  {
   for(int j=0;j<NDIVY;j++)
    mask2d0[NDIVY*i+j]=1;
  }
 if(npol==0)
   ampS('X',Axab, NDIVX, NDIVY, mask2d0, LMAX, Lrange2, MMAX, Mrange2, Nrangep, lindexp, mindexp, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, kx0, ky0, eabs, dabs, cexpX,cexpY);
 else
  ampS('Y',Axab, NDIVX, NDIVY, mask2d0, LMAX, Lrange2, MMAX, Mrange2, Nrangep, lindexp, mindexp, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, kx0, ky0, eabs, dabs, cexpX,cexpY);

 Eigen::MatrixXcd vcxx(nsourceX,nsourceY);
 Eigen::MatrixXcd abxx(nsourceX,nsourceY);
 for (int ls = -lsmaxX; ls<=lsmaxX; ls++)
 {
 for (int ms = -lsmaxY; ms<=lsmaxY; ms++)
 {
 if((pow(ls*MX/dx,2)+pow(ms*MY/dy,2))<=pow(NA/lambda,2))
  {
    for (int i = 0; i < Nrangep; i++)
   {
    if ((lindexp[i] == ls) && (mindexp[i] == ms))
    {
     vcxx(ls+lsmaxX,ms+lsmaxY)=Axvc[ls+lsmaxX][ms+lsmaxY](i);
     abxx(ls+lsmaxX,ms+lsmaxY)=Axab[ls+lsmaxX][ms+lsmaxY](i);
    } 
   }
  }
 }
 }

  Eigen::MatrixXd mask(NDIVX,NDIVY);
  Eigen::MatrixXd pattern(FDIVX,FDIVY);
  Eigen::MatrixXcd fmask(noutX,noutY);
  vector< complex<double> > fmaskx(FDIVX), fmasky(FDIVY);
  Eigen::MatrixXcd ftmp (FDIVX, noutY);
  vector<vector<Eigen::MatrixXcd>>  fampxx(nsourceX, vector<Eigen::MatrixXcd>(nsourceY));
  vector<vector<Eigen::MatrixXcd>>  ampxx(nsourceX, vector<Eigen::MatrixXcd>(nsourceY));
  vector<vector<Eigen::MatrixXcd>>  dampxx(nsourceX, vector<Eigen::MatrixXcd>(nsourceY));
  Eigen::MatrixXcd phasexx(nsourceX,nsourceY);
  for(int is=0;is<nsourceX;is++)
 {
  for(int js=0;js<nsourceY;js++)
  {
    fampxx[is][js].resize(noutX,noutY);
    ampxx[is][js].resize(noutX,noutY);
    dampxx[is][js].resize(noutX,noutY);
  }
 } 

 vector<vector<Eigen::VectorXcd>> Ax(nsourceX, vector(nsourceY, Eigen::VectorXcd(Nrange)));
 if(npol==0)
  ampS('X',Ax, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrange, lindex, mindex, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, kx0, ky0, eabs, dabs, cexpX,cexpY);
 else
  ampS('Y',Ax, NDIVX, NDIVY, mask2d, LMAX, Lrange2, MMAX, Mrange2, Nrange, lindex, mindex, FDIVX, FDIVY,
      NA, MX, MY, dx, dy, lambda, NABS, NML, lsmaxX, lsmaxY, k, kx0, ky0, eabs, dabs, cexpX,cexpY);

 for(int i=0;i<NDIVX;i++)
   for(int j=0;j<NDIVY;j++)
    mask(i,j)=mask2d[NDIVY*i+j];

  int idiv = 1./ delta + 0.000001;
  for (int i = 0; i< NDIVX; i++)
   {
    for (int ii= i*idiv; ii< (i+1)*idiv; ii++)
    {
     for(int j=0;j<NDIVY;j++)
     {
      for(int jj=j*idiv;jj<(j+1)*idiv;jj++)
       pattern(ii,jj)=mask(i,j);
     } 
    }
   }

  for (int i = 0; i < FDIVX; i++)
  {
      for (int j = 0; j < FDIVY; j++)
       fmasky[j] = pattern(i, j);
     for (int j = 0; j < noutY; j++)
     {
        int m = j - lpmaxY;
        ftmp(i, j) = fourier(m, fmasky, cexpY,FDIVY);
      }
  }

  for (int j = 0; j < noutY; j++)
 {
       for (int i = 0; i < FDIVX; i++)
                fmaskx[i] = ftmp(i,j);
     for (int i = 0; i < noutX; i++)
     {
        int l = i - lpmaxX;
        fmask(i, j) = fourier(l, fmaskx, cexpX,FDIVX);
     }
  }

 double kxs,kys,kxp,kyp;
 complex<double> phasesp;
 for(int is=0;is<nsourceX;is++)
 {
//  kxs=kx0+2.*pi*(is-lsmaxX)/dx;
  kxs=kx0;
  for(int js=0;js<nsourceY;js++)
  {
//   kys=ky0+2.*pi*(js-lsmaxY)/dy;
   kys=ky0;
   for(int ip=0;ip<noutX;ip++)
   {
     kxp=2.*pi*(ip-lpmaxX)/dx;
     for(int jp=0;jp<noutY;jp++)
     {
     kyp=2.*pi*(jp-lpmaxY)/dy;
     phasesp=exp(-zi*(kxs*kxp+kxp*kxp/2.+kys*kyp+kyp*kyp/2.)/k*z0);
     fampxx[is][js](ip,jp)=fmask(ip,jp)*phasesp*(abxx(lsmaxX,lsmaxY)-vcxx(lsmaxX,lsmaxY));
      if((ip==lpmaxX)&&(jp==lpmaxY))
      {
        fampxx[is][js](ip,jp)+=vcxx(is,js);
      }
     }
    }
   }
 }

  for(int is=0;is<nsourceX;is++)
  {
   for(int js=0;js<nsourceY;js++)
   {
    if((pow((is-lsmaxX)*MX/dx,2)+pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2))
    {
    for(int n=0;n<Nrange;n++)
    {
    int ip=lindex[n]-(is-lsmaxX)+lpmaxX;
    int jp=mindex[n]-(js-lsmaxY)+lpmaxY;
    if((pow(lindex[n]*MX/dx,2)+pow(mindex[n]*MY/dy,2))<=pow(NA/lambda,2))
     ampxx[is][js](ip,jp)=Ax[is][js](n);
    }
    }
   }
  }

  for(int is=0;is<nsourceX;is++)
  {
   for(int js=0;js<nsourceY;js++)
   {
     phasexx(is,js)=vcxx(is,js)/abs(vcxx(is,js));
 //         cout<<vcxx(is,js)<<","<<phasexx(is,js)<<endl;
//     phasexx(is,js)=fampxx[is][js](lpmaxX,lpmaxY)/abs(fampxx[is][js](lpmaxX,lpmaxY));
    } 
   }

 for(int is=0;is<nsourceX;is++)
 {
  for(int js=0;js<nsourceY;js++)
  {
   for(int ip=0;ip<noutX;ip++)
   {
    for(int jp=0;jp<noutY;jp++)
    {
      fampxx[is][js](ip,jp)=fampxx[is][js](ip,jp)/phasexx(is,js);
      if(((pow((is-lsmaxX)*MX/dx,2)
           +pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2))
          &&((pow((ip-lpmaxX+is-lsmaxX)*MX/dx,2)
           +pow((jp-lpmaxY+js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)))
     {
      ampxx[is][js](ip,jp)=ampxx[is][js](ip,jp)/phasexx(is,js);
     }
    }
   }
  }
 }

 for(int is=0;is<nsourceX;is++)
 {
  for(int js=0;js<nsourceY;js++)
  {
   for(int ip=0;ip<noutX;ip++)
   {
    for(int jp=0;jp<noutY;jp++)
    {
      if(((pow((is-lsmaxX)*MX/dx,2)
           +pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2))
       &&((pow((ip-lpmaxX+is-lsmaxX)*MX/dx,2)
           +pow((jp-lpmaxY+js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)))
     {
      dampxx[is][js](ip,jp)=ampxx[is][js](ip,jp)-fampxx[lsmaxX][lsmaxY](ip,jp);
     }
    }
   }
  }
 } 

 Eigen::MatrixXcd a0xx(noutX,noutY),axxx(noutX,noutY),ayxx(noutX,noutY);
 complex<double> f,xxf1,xxfx,xxfy,xyf1,xyfx,xyfy,yxf1,yxfx,yxfy,yyf1,yyfx,yyfy;
 complex<double> c1,cx,cy,cx2,cy2,cxy,cd;
 for(int n=0;n<ninput;n++) 
 {
   int ip=linput[n]+lpmaxX;
   int jp=minput[n]+lpmaxY;
   double lp = ip - lpmaxX;
   double mp = jp - lpmaxY;
   xxf1=0.;xxfx=0.;xxfy=0.;
   c1=0.;cx=0.;cy=0.;cx2=0.;cy2=0.;cxy=0.;
   for(int is=0;is<nsourceX;is++)
   {
    for(int js=0;js<nsourceY;js++) 
    {
     double ls = is - lsmaxX+lp/2.;
      double ms = js - lsmaxY+mp/2.;
     if(((pow((is-lsmaxX)*MX/dx,2)
                  +pow((js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2))
       &&((pow((ip-lpmaxX+is-lsmaxX)*MX/dx,2)
                  +pow((jp-lpmaxY+js-lsmaxY)*MY/dy,2))<=pow(NA/lambda,2)))
       {  
        f=dampxx[is][js](ip,jp);
        xxf1+=f;xxfx+=ls*f;xxfy+=ms*f;
        c1+=1;cx+=ls;cy+=ms;cx2+=ls*ls;cy2+=ms*ms;cxy+=ls*ms;
       }
     }
   }
   if(xinput[n]==1)
   {
     cd=c1*cx2*cy2+2.*cx*cxy*cy-c1*cxy*cxy-cx2*cy*cy-cy2*cx*cx;
     a0xx(ip,jp)=1./cd*((cx2*cy2-cxy*cxy)*xxf1+(-cx*cy2+cy*cxy)*xxfx+(cx*cxy-cy*cx2)*xxfy);
     axxx(ip,jp)=1./cd*((-cx*cy2+cy*cxy)*xxf1+(c1*cy2-cy*cy)*xxfx+(-c1*cxy+cx*cy)*xxfy);
     ayxx(ip,jp)=1./cd*((cx*cxy-cy*cx2)*xxf1+(-c1*cxy+cx*cy)*xxfx+(c1*cx2-cx*cx)*xxfy);
   }
   else
   {
     a0xx(ip,jp)=xxf1/c1;
     axxx(ip,jp)=0.;
     ayxx(ip,jp)=0.;
   }
 }  
 for(int n=0;n<ninput;n++) 
 {
  int ip=linput[n]+lpmaxX;
  int jp=minput[n]+lpmaxY;

 if(npol==0)
  ofsxx << real(a0xx(ip,jp)) << "," ;
// else
//  ofsyy << real(a0xx(ip,jp)) << "," ;
 }
 if(npol==0)
  ofsxx<<endl;
// else
//  ofsyy<<endl;
 for(int n=0;n<ninput;n++) 
  {
   int ip=linput[n]+lpmaxX;
   int jp=minput[n]+lpmaxY;
 if(npol==0)
   ofsxx << imag(a0xx(ip,jp)) << "," ;
// else
//   ofsyy << imag(a0xx(ip,jp)) << "," ;
  }
 if(npol==0)
  ofsxx<<endl;
// else
//  ofsyy<<endl;
 for(int n=0;n<ninput;n++) 
 {
  if(xinput[n]==1)
  {
   int ip=linput[n]+lpmaxX;
   int jp=minput[n]+lpmaxY;
 if(npol==0)
   ofsxx << real(axxx(ip,jp)) << "," ;
// else
//   ofsyy << real(axxx(ip,jp)) << "," ;
  }
 }
 if(npol==0)
  ofsxx<<endl;
// else
//  ofsyy<<endl;
 for(int n=0;n<ninput;n++) 
 {
  if(xinput[n]==1)
  {
   int ip=linput[n]+lpmaxX;
   int jp=minput[n]+lpmaxY;
 if(npol==0)
   ofsxx << imag(axxx(ip,jp)) << "," ;
// else
//   ofsyy << imag(axxx(ip,jp)) << "," ;
  }
 }
 if(npol==0)
  ofsxx<<endl;
// else
//  ofsyy<<endl;
 for(int n=0;n<ninput;n++) 
 {
  if(xinput[n]==1)
  {
   int ip=linput[n]+lpmaxX;
   int jp=minput[n]+lpmaxY;
 if(npol==0)
   ofsxx << real(ayxx(ip,jp)) << "," ;
// else
//   ofsyy << real(ayxx(ip,jp)) << "," ;
  }
 }
 if(npol==0)
  ofsxx<<endl;
// else
//  ofsyy<<endl;
 for(int n=0;n<ninput;n++) 
 {
  if(xinput[n]==1)
  {
   int ip=linput[n]+lpmaxX;
   int jp=minput[n]+lpmaxY;
 if(npol==0)
   ofsxx << imag(ayxx(ip,jp)) << "," ;
// else
//   ofsyy << imag(ayxx(ip,jp)) << "," ;
  }
 }
 if(npol==0)
  ofsxx<<endl;
// else
//  ofsyy<<endl;
}

 if(nsample==0)
 {
 now = std::chrono::system_clock::now(); 
 elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now-start).count();
// cout<<"elapsed time (s) "<<elapsed/1000.<<endl;
 }
}
 magma_finalize();
 return 0;
}

