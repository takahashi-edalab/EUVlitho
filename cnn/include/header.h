complex<double> fourier(int l,vector<complex<double> >& f,vector< complex<double> >& cexp,int FDIV);
void matinv(int N, Eigen::MatrixXcd& A,Eigen::MatrixXcd& Ainv);
void matproduct(int N, Eigen::MatrixXcd& A,Eigen::MatrixXcd& B,Eigen::MatrixXcd& C);

void multilayerS(char polar,int Nrange,int NML,double k,const vector<double>& kxplus,
      const vector<double>& kyplus,const vector<double>& kxy2,complex<double> esio2,
      complex<double> emo,complex<double> esi, complex<double> emosi2,complex<double> eru,
      complex<double> erusi,double dmo, double dmosi, double dsi, double dsimo, double dru,double dsiru,
      Eigen::SparseMatrix<complex< double >>& URUU,Eigen::SparseMatrix<complex< double >>& URUB)
 {
  complex<double> zi (0., 1.);
  vector< complex<double> > alsio2(Nrange),almo(Nrange),alsi(Nrange),almosi2(Nrange),
      alru(Nrange),alrusi(Nrange);
  typedef Eigen::SparseMatrix<complex< double >> Sparse;
  Sparse TMOUL(Nrange,Nrange),TMOUR(Nrange,Nrange),TMOBL(Nrange,Nrange),
      TMOBR(Nrange,Nrange);
  Sparse TMOSIUL(Nrange,Nrange),TMOSIUR(Nrange,Nrange),TMOSIBL(Nrange,Nrange),
      TMOSIBR(Nrange,Nrange);
  Sparse TSIUL(Nrange,Nrange),TSIUR(Nrange,Nrange),TSIBL(Nrange,Nrange),
      TSIBR(Nrange,Nrange);
  Sparse TSIRUUL(Nrange,Nrange),TSIRUUR(Nrange,Nrange),TSIRUBL(Nrange,Nrange),
      TSIRUBR(Nrange,Nrange);
  Sparse TSIMOUL(Nrange,Nrange),TSIMOUR(Nrange,Nrange),TSIMOBL(Nrange,Nrange),
      TSIMOBR(Nrange,Nrange);
  Sparse TRUUL(Nrange,Nrange),TRUUR(Nrange,Nrange),TRUBL(Nrange,Nrange),
      TRUBR(Nrange,Nrange);
  Sparse TNU(Nrange,Nrange),TNB(Nrange,Nrange);
  Sparse UU(Nrange,Nrange),UB(Nrange,Nrange),UU_org(Nrange,Nrange),
      UB_org(Nrange,Nrange);
  typedef Eigen::Triplet<complex< double >> Tr;
  vector<Tr> data(Nrange),Cj(Nrange),Cjp(Nrange);

 for (int i=0;i<Nrange;i++)
  {
   alsio2[i]=sqrt(k*k*esio2-kxy2[i]);
   almo[i]=sqrt(k*k*emo-kxy2[i]);
   alsi[i]=sqrt(k*k*esi-kxy2[i]);
   almosi2[i]=sqrt(k*k*emosi2-kxy2[i]);
   alru[i]=sqrt(k*k*eru-kxy2[i]);
   alrusi[i]=sqrt(k*k*erusi-kxy2[i]);
  }
 for (int i=0;i<Nrange;i++)
 {
  data[i]=Tr(i,i,1.);
  }
 Cjp=data; 
 if(polar=='X')
 {
  for (int i=0;i<Nrange;i++)
  {
   data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/emosi2)/(k-kxplus[i]*kxplus[i]/k/emo));
  }
  Cj=data;
 }
 else if(polar=='Y')
 {
  for (int i=0;i<Nrange;i++)
  {
    data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/emosi2)/(k-kyplus[i]*kyplus[i]/k/emo));
  }
  Cj=data;
 } 
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almo[i]*dmo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+almosi2[i]/almo[i]*Cjp[i].value())/expz);
 }
 TMOUL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {
  complex<double> expz=exp(zi*almo[i]*dmo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-almosi2[i]/almo[i]*Cjp[i].value())/expz);
 }
 TMOUR.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almo[i]*dmo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-almosi2[i]/almo[i]*Cjp[i].value())*expz);
 }
 TMOBL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {
  complex<double> expz=exp(zi*almo[i]*dmo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+almosi2[i]/almo[i]*Cjp[i].value())*expz);
 }
 TMOBR.setFromTriplets(data.begin(),data.end());

 if(polar=='X')
 {
 for (int i=0;i<Nrange;i++)
 {
  data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/emo)/(k-kxplus[i]*kxplus[i]/k/emosi2));
 }
 Cj=data;
 }
 else if(polar=='Y')
 {
 for (int i=0;i<Nrange;i++)
 {
  data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/emo)/(k-kyplus[i]*kyplus[i]/k/emosi2));
 }
 Cj=data;
 }
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dmosi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+almo[i]/almosi2[i]*Cjp[i].value())/expz);
 }
 TMOSIUL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dmosi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-almo[i]/almosi2[i]*Cjp[i].value())/expz);
 }
 TMOSIUR.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dmosi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-almo[i]/almosi2[i]*Cjp[i].value())*expz);
 }
 TMOSIBL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dmosi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+almo[i]/almosi2[i]*Cjp[i].value())*expz);
 }
 TMOSIBR.setFromTriplets(data.begin(),data.end());

 if(polar=='X')
 {
  for (int i=0;i<Nrange;i++)
  {
   data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/emosi2)/(k-kxplus[i]*kxplus[i]/k/esi));
  }
  Cj=data;
  }
 else if(polar=='Y')
 {
  for (int i=0;i<Nrange;i++)
  {
   data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/emosi2)/(k-kyplus[i]*kyplus[i]/k/esi));
  }
  Cj=data;
 }
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alsi[i]*dsi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+almosi2[i]/alsi[i]*Cjp[i].value())/expz);
 }
 TSIUL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alsi[i]*dsi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-almosi2[i]/alsi[i]*Cjp[i].value())/expz);
 }
 TSIUR.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alsi[i]*dsi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-almosi2[i]/alsi[i]*Cjp[i].value())*expz);
 }
 TSIBL.setFromTriplets(data.begin(),data.end());

 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alsi[i]*dsi);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+almosi2[i]/alsi[i]*Cjp[i].value())*expz);
 }
 TSIBR.setFromTriplets(data.begin(),data.end());

 if(polar=='X')
 {
  for (int i=0;i<Nrange;i++)
  {
   data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/esi)/(k-kxplus[i]*kxplus[i]/k/emosi2));
  }
  Cj=data;
 }
 else if(polar=='Y')
 {
  for (int i=0;i<Nrange;i++)
  {
   data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/esi)/(k-kyplus[i]*kyplus[i]/k/emosi2));
  }
  Cj=data;
 }
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dsimo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alsi[i]/almosi2[i]*Cjp[i].value())/expz);
 }
 TSIMOUL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dsimo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alsi[i]/almosi2[i]*Cjp[i].value())/expz);
 }
 TSIMOUR.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dsimo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alsi[i]/almosi2[i]*Cjp[i].value())*expz);
 }
 TSIMOBL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almosi2[i]*dsimo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alsi[i]/almosi2[i]*Cjp[i].value())*expz);
 }
 TSIMOBR.setFromTriplets(data.begin(),data.end());

 if(polar=='X')
 {
 for (int i=0;i<Nrange;i++)
 {
   data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/esi)/(k-kxplus[i]*kxplus[i]/k/erusi));
 }
 Cj=data;
 }
 else if(polar=='Y')
 {
 for (int i=0;i<Nrange;i++)
 {
   data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/esi)/(k-kyplus[i]*kyplus[i]/k/erusi));
 }
 Cj=data;
 }
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alrusi[i]*dsiru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alsi[i]/alrusi[i]*Cjp[i].value())/expz);
 }
 TSIRUUL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alrusi[i]*dsiru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alsi[i]/alrusi[i]*Cjp[i].value())/expz);
 }
 TSIRUUR.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alrusi[i]*dsiru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alsi[i]/alrusi[i]*Cjp[i].value())*expz);
 }
 TSIRUBL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alrusi[i]*dsiru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alsi[i]/alrusi[i]*Cjp[i].value())*expz);
 }
 TSIRUBR.setFromTriplets(data.begin(),data.end());

 if(polar=='X')
 {
  for (int i=0;i<Nrange;i++)
  {
   data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/erusi)/(k-kxplus[i]*kxplus[i]/k/eru));
  }
  Cj=data;
 }
 else if(polar=='Y')
 {
  for (int i=0;i<Nrange;i++)
  {
  data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/erusi)/(k-kyplus[i]*kyplus[i]/k/eru));
  }
  Cj=data;
 }
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alru[i]*dru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alrusi[i]/alru[i]*Cjp[i].value())/expz);
 }
 TRUUL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alru[i]*dru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alrusi[i]/alru[i]*Cjp[i].value())/expz);
 }
 TRUUR.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alru[i]*dru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alrusi[i]/alru[i]*Cjp[i].value())*expz);
 }
 TRUBL.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*alru[i]*dru);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alrusi[i]/alru[i]*Cjp[i].value())*expz);
 }
 TRUBR.setFromTriplets(data.begin(),data.end());

 if(polar=='X')
 {
 for (int i=0;i<Nrange;i++)
 {
  data[i]=Tr(i,i,(k-kxplus[i]*kxplus[i]/k/esio2)/(k-kxplus[i]*kxplus[i]/k/emo));
 }
 Cj=data;
 }
 else if(polar=='Y')
 {
 for (int i=0;i<Nrange;i++)
 {
  data[i]=Tr(i,i,(k-kyplus[i]*kyplus[i]/k/esio2)/(k-kyplus[i]*kyplus[i]/k/emo));
 }
 Cj=data;
 }
 for (int i=0;i<Nrange;i++)
 {  
 complex<double> expz=exp(zi*almo[i]*dmo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()+alsio2[i]/almo[i]*Cjp[i].value())/expz);
 }
 TNU.setFromTriplets(data.begin(),data.end());
 for (int i=0;i<Nrange;i++)
 {  
  complex<double> expz=exp(zi*almo[i]*dmo);
  data[i]=Tr(i,i,1./2.*(Cj[i].value()-alsio2[i]/almo[i]*Cjp[i].value())*expz);
 }
 TNB.setFromTriplets(data.begin(),data.end());

 UU=TNU;
 UB=TNB;
 
 for(int i=NML-1;i>=0;i--)
 {
   if(i<NML-1)
    {UU_org=UU;
     UB_org=UB;
     UU=TMOUL*UU_org+TMOUR*UB_org;
     UB=TMOBL*UU_org+TMOBR*UB_org;
    } 

   UU_org=UU;
   UB_org=UB;
   UU=TMOSIUL*UU_org+TMOSIUR*UB_org;
   UB=TMOSIBL*UU_org+TMOSIBR*UB_org;

   UU_org=UU;
   UB_org=UB;
   UU=TSIUL*UU_org+TSIUR*UB_org;
   UB=TSIBL*UU_org+TSIBR*UB_org;

   if(i>0)
   {
    UU_org=UU;
    UB_org=UB;
    UU=TSIMOUL*UU_org+TSIMOUR*UB_org;
    UB=TSIMOBL*UU_org+TSIMOBR*UB_org;
   }
   else
    {
    UU_org=UU;
    UB_org=UB;
    UU=TSIRUUL*UU_org+TSIRUUR*UB_org;
    UB=TSIRUBL*UU_org+TSIRUBR*UB_org;
   }
 }
 
 URUU=TRUUL*UU+TRUUR*UB;
 URUB=TRUBL*UU+TRUBR*UB;
}

void absorberS(char polar,int LMAX,int MMAX,int Nrange,const vector<int>& lindex,
      const vector<int>& mindex,double k,const vector<double>& kxplus,
      const vector<double>& kyplus,const vector<double>& kxy2,const Eigen::MatrixXcd& eps1,
      const Eigen::MatrixXcd& eta1,const Eigen::MatrixXcd& zeta1,const Eigen::MatrixXcd& sigma1,
      double dabs1,vector< complex<double> >& al2,vector<Eigen::VectorXcd>& br2,
      Eigen::MatrixXcd& B2,Eigen::MatrixXcd& U2U,Eigen::MatrixXcd& U2B,
      vector< complex<double> >& al1,vector<Eigen::VectorXcd>& br1,
      Eigen::MatrixXcd& B1,Eigen::MatrixXcd& U1U,Eigen::MatrixXcd& U1B)
{
 complex<double> zi (0., 1.);
 magma_vec_t  jobvl,jobvr;
 magmaDoubleComplex *h_R, *VL, *VR, *h_work, *w1,*w2;
 double *rwork;
 magma_int_t N, n2, lda, nb, lwork, lwork2, info,ngpu;
 jobvl = MagmaNoVec;
 jobvr = MagmaVec; 
 N=Nrange;
 lda=N;
 n2    = lda*N;
 nb    = magma_get_zgehrd_nb(N);
 ngpu=magma_num_gpus();
 lwork = N*(1 + 2*nb+ngpu*nb);
 lwork2 = max( lwork, N*(5 + 2*N) );            
 magma_zmalloc_cpu( &w1,     N );
 magma_zmalloc_cpu( &w2,     N );
 magma_dmalloc_cpu( &rwork,  2*N );
 magma_zmalloc_pinned( &h_R,    n2 );
 magma_zmalloc_pinned( &VL,     n2 );
 magma_zmalloc_pinned( &VR,     n2 );
 magma_zmalloc_pinned( &h_work, lwork2 );

 vector< complex<double> > al1sq(Nrange);
 {
  Eigen::MatrixXcd D(Nrange,Nrange);
   #pragma omp parallel for
  for(int i=0;i<Nrange;i++)
  {int l=lindex[i];
   int m=mindex[i];
   for(int j=0;j<Nrange;j++)
   {int llp=l-lindex[j]+2*LMAX;
    int mmp=m-mindex[j]+2*MMAX;
        if(polar=='X')
       D(i,j)=k*k*eps1(llp,mmp)-zi*eta1(llp,mmp)*kxplus[j];
     else if(polar=='Y')
       D(i,j)=k*k*eps1(llp,mmp)-zi*zeta1(llp,mmp)*kyplus[j];
    if(i==j)
    {
      D(i,i)=D(i,i)-kxy2[i];
    }
   }
  }

/*
 Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigensolver(D);
#pragma omp parallel for
  for(int i=0;i<Nrange;i++)
 {
  al1sq[i]=eigensolver.eigenvalues()(i);
  for(int j=0;j<Nrange;j++)
  {
   br1[j](i)=eigensolver.eigenvectors()(i,j);
  }
 }
}
*/

  #pragma omp parallel for
  for(int i=0;i<Nrange;i++)
  {
   for(int j=0;j<Nrange;j++)
   {
     h_R[i+j*N]=MAGMA_Z_MAKE(real(D(i,j)),imag(D(i,j)));
   }
  }
 }

 magma_zgeev_m( jobvl, jobvr,N, h_R, lda, w1,VL, lda, VR, lda,h_work, lwork, rwork, &info );
 if(info!=0)   cout<<"zgeev "<<info<<endl;

 #pragma omp parallel for
 for(int i=0;i<Nrange;i++)
 {
  al1sq[i]=MAGMA_Z_REAL(w1[i])+zi*MAGMA_Z_IMAG(w1[i]);
  for(int j=0;j<Nrange;j++)
  {
   br1[i](j)=MAGMA_Z_REAL(VR[i*N+j])+zi*MAGMA_Z_IMAG(VR[i*N+j]);
  }
 }

 magma_free_cpu(w1);
 magma_free_cpu(w2);
 magma_free_cpu(rwork);
 magma_free_pinned(h_R);
 magma_free_pinned(VL);
 magma_free_pinned(VR);
 magma_free_pinned(h_work);

 for (int i = 0; i<Nrange; i++)
 {
  al1[i] = sqrt(al1sq[i]);
 }

 Eigen::MatrixXcd Cjp(Nrange,Nrange);
 {
  Eigen::MatrixXcd FG(Nrange, Nrange);
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
    FG(i, j) = br1[j](i);
  }

  Eigen::MatrixXcd FGinv(Nrange, Nrange);
  matinv(Nrange, FG,FGinv);

/*
 #pragma omp parallel for
 for (int i = 0; i<Nrange; i++)
 {
   for (int j = 0; j < Nrange; j++)
	 bl1[i](j) = FGinv(i, j);
 }
*/
/*
 for(int i=0;i<Nrange;i++)
 {
  al1[i]=sqrt(al1sq[i]);
  complex<double> norm1;
  norm1=bl1[i]*br1[i];
  for(int j=0;j<Nrange;j++)
  bl1[i](j)=bl1[i](j)/norm1;
 }
*/
/*
#pragma omp parallel for
for (int i=0;i<Nrange;i++)
{
 for(int j=0;j<Nrange;j++)
 {
  Cjp(i,j)=bl1[i]*br2[j]; 
 }
}
*/

  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
    FG(i, j) = br2[j](i);
  }

//Cjp=FGinv*FG;
  matproduct(Nrange,FGinv,FG,Cjp);
 }

 if(polar=='X')
 {
  {
  Eigen::MatrixXcd BR(Nrange, Nrange);
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
   BR(i, j) = br1[j](i);
  }
  Eigen::MatrixXcd Sigma(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
  {
   int l=lindex[i];
   int m=mindex[i];
   for(int ip=0;ip<Nrange;ip++)
   {
    int llp=l-lindex[ip]+2*LMAX;
    int mmp=m-mindex[ip]+2*MMAX;
    Sigma(i,ip)=sigma1(llp,mmp)*kxplus[ip];
   }
  }
//B1=Sigma*BR;
   matproduct(Nrange,Sigma,BR,B1);
 }
 Eigen::MatrixXcd SUM(Nrange,Nrange);
 SUM=B1;
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  for(int j=0;j<Nrange;j++)
  {
   B1(i,j)=zi*(k*br1[j](i)-kxplus[i]/k*SUM(i,j));
  }
 }
 }
 else if(polar=='Y')
 {
  {
  Eigen::MatrixXcd BR(Nrange, Nrange);
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
   BR(i, j) = br1[j](i);
  }
 Eigen::MatrixXcd Sigma(Nrange,Nrange);
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  int l=lindex[i];
  int m=mindex[i];
   for(int ip=0;ip<Nrange;ip++)
   {
    int llp=l-lindex[ip]+2*LMAX;
    int mmp=m-mindex[ip]+2*MMAX;
    Sigma(i,ip)=sigma1(llp,mmp)*kyplus[ip];
   }
 }
//B1=Sigma*BR;
  matproduct(Nrange,Sigma,BR,B1);
 }
 Eigen::MatrixXcd SUM(Nrange,Nrange);
 SUM=B1;
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  for(int j=0;j<Nrange;j++)
  {
   B1(i,j)=zi*(k*br1[j](i)-kyplus[i]/k*SUM(i,j));
  }
 }
 }
 Eigen::MatrixXcd Cj(Nrange,Nrange);
 matinv(Nrange, B1,Cj);
// Cj=Cj*B2;
 matproduct(Nrange,Cj,B2,Cj);

/*
{
 Eigen::MatrixXcd T1UL(Nrange,Nrange),T1UR(Nrange,Nrange),T1BL(Nrange,Nrange),T1BR(Nrange,Nrange);
  vector<complex<double>> gamma(Nrange);
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
  gamma[i]=exp(zi*al1[i]*dabs1);

 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  for(int j=0;j<Nrange;j++)
  {
  complex<double> ctmp,atmp;
   ctmp=1./2.*Cj(i,j);
   atmp=1./2.*al2[j]/al1[i]*Cjp(i,j);
   T1UL(i,j)=(ctmp+atmp)/gamma[i];
   T1UR(i,j)=(ctmp-atmp)/gamma[i];
   T1BL(i,j)=(ctmp-atmp)*gamma[i];
   T1BR(i,j)=(ctmp+atmp)*gamma[i];
  }
 }
 U1U=T1UL*U2U+T1UR*U2B;
 U1B=T1BL*U2U+T1BR*U2B;
}
*/

 {
  Eigen::MatrixXcd T1UL(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1UL(i,j)=1./2.*(Cj(i,j)+al2[j]/al1[i]*Cjp(i,j))/gamma;
  }
 }
  matproduct(Nrange,T1UL,U2U,U1U);
//U1U=T1UL*U2U;
 }
 { 
  Eigen::MatrixXcd T1UR(Nrange,Nrange);
  Eigen::MatrixXcd C(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1UR(i,j)=1./2.*(Cj(i,j)-al2[j]/al1[i]*Cjp(i,j))/gamma;
  }
 }
  matproduct(Nrange,T1UR,U2B,C);
  U1U+=C;
//U1U+=T1UR*U2B;
 }
 {
  Eigen::MatrixXcd T1BL(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1BL(i,j)=1./2.*(Cj(i,j)-al2[j]/al1[i]*Cjp(i,j))*gamma;
  }
 }
  matproduct(Nrange,T1BL,U2U,U1B);
//U1B=T1BL*U2U;
 }
 {
  Eigen::MatrixXcd T1BR(Nrange,Nrange);
  Eigen::MatrixXcd C(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1BR(i,j)=1./2.*(Cj(i,j)+al2[j]/al1[i]*Cjp(i,j))*gamma;
  }
 }
  matproduct(Nrange,T1BR,U2B,C);
  U1B+=C;
//U1B+=T1BR*U2B;
 }
}

void absorberS0(char polar,int LMAX,int MMAX,int Nrange,const vector<int>& lindex,
      const vector<int>& mindex,double k,const vector<double>& kxplus,
      const vector<double>& kyplus,const vector<double>& kxy2,const Eigen::MatrixXcd& eps1,
      const Eigen::MatrixXcd& eta1,const Eigen::MatrixXcd& zeta1,const Eigen::MatrixXcd& sigma1,
      double dabs1,vector< complex<double> >& al2,Eigen::SparseMatrix<complex< double >>& br2,
      Eigen::SparseMatrix<complex< double >>& B2,Eigen::SparseMatrix<complex< double >>& U2U,
      Eigen::SparseMatrix<complex< double >>& U2B,
      vector< complex<double> >& al1,vector<Eigen::VectorXcd>& br1,
      Eigen::MatrixXcd& B1,Eigen::MatrixXcd& U1U,Eigen::MatrixXcd& U1B)
{
 complex<double> zi (0., 1.);
 magma_vec_t  jobvl,jobvr;
 magmaDoubleComplex *h_R, *VL, *VR, *h_work, *w1,*w2;
 double *rwork;
 magma_int_t N, n2, lda, nb, lwork, lwork2, info,ngpu;
 jobvl = MagmaNoVec;
 jobvr = MagmaVec; 
 N=Nrange;
 lda=N;
 n2    = lda*N;
 nb    = magma_get_zgehrd_nb(N);
 ngpu=magma_num_gpus();
 lwork = N*(1 + 2*nb+ngpu*nb);
 lwork2 = max( lwork, N*(5 + 2*N) );            
 magma_zmalloc_cpu( &w1,     N );
 magma_zmalloc_cpu( &w2,     N );
 magma_dmalloc_cpu( &rwork,  2*N );
 magma_zmalloc_pinned( &h_R,    n2 );
 magma_zmalloc_pinned( &VL,     n2 );
 magma_zmalloc_pinned( &VR,     n2 );
 magma_zmalloc_pinned( &h_work, lwork2 );

 vector< complex<double> > al1sq(Nrange);
 {
  Eigen::MatrixXcd D(Nrange,Nrange);
   #pragma omp parallel for
  for(int i=0;i<Nrange;i++)
  {int l=lindex[i];
   int m=mindex[i];
   for(int j=0;j<Nrange;j++)
   {int llp=l-lindex[j]+2*LMAX;
    int mmp=m-mindex[j]+2*MMAX;
        if(polar=='X')
       D(i,j)=k*k*eps1(llp,mmp)-zi*eta1(llp,mmp)*kxplus[j];
     else if(polar=='Y')
       D(i,j)=k*k*eps1(llp,mmp)-zi*zeta1(llp,mmp)*kyplus[j];
    if(i==j)
    {
      D(i,i)=D(i,i)-kxy2[i];
    }
   }
  }

/*
 Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigensolver(D);
#pragma omp parallel for
  for(int i=0;i<Nrange;i++)
 {
  al1sq[i]=eigensolver.eigenvalues()(i);
  for(int j=0;j<Nrange;j++)
  {
   br1[j](i)=eigensolver.eigenvectors()(i,j);
  }
 }
}
*/

  #pragma omp parallel for
  for(int i=0;i<Nrange;i++)
  {
   for(int j=0;j<Nrange;j++)
   {
     h_R[i+j*N]=MAGMA_Z_MAKE(real(D(i,j)),imag(D(i,j)));
   }
  }
 }

 magma_zgeev_m( jobvl, jobvr,N, h_R, lda, w1,VL, lda, VR, lda,h_work, lwork, rwork, &info );
 if(info!=0)   cout<<"zgeev "<<info<<endl;

 #pragma omp parallel for
 for(int i=0;i<Nrange;i++)
 {
  al1sq[i]=MAGMA_Z_REAL(w1[i])+zi*MAGMA_Z_IMAG(w1[i]);
  for(int j=0;j<Nrange;j++)
  {
   br1[i](j)=MAGMA_Z_REAL(VR[i*N+j])+zi*MAGMA_Z_IMAG(VR[i*N+j]);
  }
 }

 magma_free_cpu(w1);
 magma_free_cpu(w2);
 magma_free_cpu(rwork);
 magma_free_pinned(h_R);
 magma_free_pinned(VL);
 magma_free_pinned(VR);
 magma_free_pinned(h_work);

 for (int i = 0; i<Nrange; i++)
 {
  al1[i] = sqrt(al1sq[i]);
 }

 Eigen::MatrixXcd Cjp(Nrange,Nrange);
 {
  Eigen::MatrixXcd FG(Nrange, Nrange);
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
    FG(i, j) = br1[j](i);
  }

  Eigen::MatrixXcd FGinv(Nrange, Nrange);
  matinv(Nrange, FG,FGinv);

/*
 #pragma omp parallel for
 for (int i = 0; i<Nrange; i++)
 {
   for (int j = 0; j < Nrange; j++)
	 bl1[i](j) = FGinv(i, j);
 }
*/
/*
 for(int i=0;i<Nrange;i++)
 {
  al1[i]=sqrt(al1sq[i]);
  complex<double> norm1;
  norm1=bl1[i]*br1[i];
  for(int j=0;j<Nrange;j++)
  bl1[i](j)=bl1[i](j)/norm1;
 }
*/
/*
#pragma omp parallel for
for (int i=0;i<Nrange;i++)
{
 for(int j=0;j<Nrange;j++)
 {
  Cjp(i,j)=bl1[i]*br2[j]; 
 }
}
*/
/*
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
    FG(i, j) = br2[j](i);
  }

//Cjp=FGinv*FG;
  matproduct(Nrange,FGinv,FG,Cjp);
*/
//  Cjp=FGinv*br2;
  Cjp=FGinv;
 }

 if(polar=='X')
 {
  {
  Eigen::MatrixXcd BR(Nrange, Nrange);
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
   BR(i, j) = br1[j](i);
  }
  Eigen::MatrixXcd Sigma(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
  {
   int l=lindex[i];
   int m=mindex[i];
   for(int ip=0;ip<Nrange;ip++)
   {
    int llp=l-lindex[ip]+2*LMAX;
    int mmp=m-mindex[ip]+2*MMAX;
    Sigma(i,ip)=sigma1(llp,mmp)*kxplus[ip];
   }
  }
//B1=Sigma*BR;
   matproduct(Nrange,Sigma,BR,B1);
 }
 Eigen::MatrixXcd SUM(Nrange,Nrange);
 SUM=B1;
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  for(int j=0;j<Nrange;j++)
  {
   B1(i,j)=zi*(k*br1[j](i)-kxplus[i]/k*SUM(i,j));
  }
 }
 }
 else if(polar=='Y')
 {
  {
  Eigen::MatrixXcd BR(Nrange, Nrange);
  #pragma omp parallel for
  for (int i = 0; i<Nrange; i++)
  {
   for (int j = 0; j < Nrange; j++)
   BR(i, j) = br1[j](i);
  }
 Eigen::MatrixXcd Sigma(Nrange,Nrange);
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  int l=lindex[i];
  int m=mindex[i];
   for(int ip=0;ip<Nrange;ip++)
   {
    int llp=l-lindex[ip]+2*LMAX;
    int mmp=m-mindex[ip]+2*MMAX;
    Sigma(i,ip)=sigma1(llp,mmp)*kyplus[ip];
   }
 }
//B1=Sigma*BR;
  matproduct(Nrange,Sigma,BR,B1);
 }
 Eigen::MatrixXcd SUM(Nrange,Nrange);
 SUM=B1;
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  for(int j=0;j<Nrange;j++)
  {
   B1(i,j)=zi*(k*br1[j](i)-kyplus[i]/k*SUM(i,j));
  }
 }
 }
 Eigen::MatrixXcd Cj(Nrange,Nrange);
 matinv(Nrange, B1,Cj);
 Cj=Cj*B2;
// matproduct(Nrange,Cj,B2,Cj);

/*
{
 Eigen::MatrixXcd T1UL(Nrange,Nrange),T1UR(Nrange,Nrange),T1BL(Nrange,Nrange),T1BR(Nrange,Nrange);
  vector<complex<double>> gamma(Nrange);
 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
  gamma[i]=exp(zi*al1[i]*dabs1);

 #pragma omp parallel for
 for (int i=0;i<Nrange;i++)
 {
  for(int j=0;j<Nrange;j++)
  {
  complex<double> ctmp,atmp;
   ctmp=1./2.*Cj(i,j);
   atmp=1./2.*al2[j]/al1[i]*Cjp(i,j);
   T1UL(i,j)=(ctmp+atmp)/gamma[i];
   T1UR(i,j)=(ctmp-atmp)/gamma[i];
   T1BL(i,j)=(ctmp-atmp)*gamma[i];
   T1BR(i,j)=(ctmp+atmp)*gamma[i];
  }
 }
 U1U=T1UL*U2U+T1UR*U2B;
 U1B=T1BL*U2U+T1BR*U2B;
}
*/

 {
  Eigen::MatrixXcd T1UL(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1UL(i,j)=1./2.*(Cj(i,j)+al2[j]/al1[i]*Cjp(i,j))/gamma;
  }
 }
//  matproduct(Nrange,T1UL,U2U,U1U);
  U1U=T1UL*U2U;
 }
 { 
  Eigen::MatrixXcd T1UR(Nrange,Nrange);
  Eigen::MatrixXcd C(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1UR(i,j)=1./2.*(Cj(i,j)-al2[j]/al1[i]*Cjp(i,j))/gamma;
  }
 }
//  matproduct(Nrange,T1UR,U2B,C);
//  U1U+=C;
  U1U+=T1UR*U2B;
 }
 {
  Eigen::MatrixXcd T1BL(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1BL(i,j)=1./2.*(Cj(i,j)-al2[j]/al1[i]*Cjp(i,j))*gamma;
  }
 }
//  matproduct(Nrange,T1BL,U2U,U1B);
  U1B=T1BL*U2U;
 }
 {
  Eigen::MatrixXcd T1BR(Nrange,Nrange);
  Eigen::MatrixXcd C(Nrange,Nrange);
  #pragma omp parallel for
  for (int i=0;i<Nrange;i++)
 {
  complex<double> gamma=exp(zi*al1[i]*dabs1);
  for(int j=0;j<Nrange;j++)
  {
   T1BR(i,j)=1./2.*(Cj(i,j)+al2[j]/al1[i]*Cjp(i,j))*gamma;
  }
 }
//  matproduct(Nrange,T1BR,U2B,C);
//  U1B+=C;
  U1B+=T1BR*U2B;
 }
}

void maskamp(int FDIVX,int FDIVY, int NDIVX, int NDIVY, int* mask2d,
 complex<double> ampta, complex<double> ampvc, Eigen::MatrixXcd& famp,
 int Lrange2, int LMAX, int Mrange2, int MMAX, vector< complex<double> >& cexpX,
 vector< complex<double> >& cexpY)
{ 
  int meshX, meshY;
  meshX=FDIVX/NDIVX;
  meshY=FDIVY/NDIVY;
  Eigen::MatrixXcd pattern(FDIVX,FDIVY);
  for (int i = 0; i < FDIVX; i++)
  {
    int ii=i/meshX;
    for (int j = 0; j < FDIVY; j++)
    {
    int jj=j/meshY;
    if(mask2d[NDIVY*ii+jj]==1)
//    if(mask2d[ii][jj]==1)
      pattern(i, j) = ampta;
    else 
      pattern(i, j) = ampvc;
    }
  }
  vector< complex<double> > ampx(FDIVX), ampy(FDIVY);
  Eigen::MatrixXcd ftmp (FDIVX, Mrange2);
  for (int i = 0; i < FDIVX; i++)
  {
    for (int j = 0; j < FDIVY; j++)
        ampy[j] = pattern(i, j);
    for (int j = 0; j < Mrange2; j++)
  {
     int m = j - 2 * MMAX;
     ftmp(i, j) = fourier(m, ampy, cexpY, FDIVY);
  }
 }
 for (int j = 0; j < Mrange2; j++)
 {
   for (int i = 0; i < FDIVX; i++)
      ampx[i] = ftmp(i,j);
  for (int i = 0; i < Lrange2; i++)
  {
     int l = i - 2 *LMAX;
     famp(i, j) = fourier(l, ampx, cexpX, FDIVX);
  }
 }
}

void ampS(char polar, vector<vector<Eigen::VectorXcd>>& Ax, int NDIVX, int NDIVY,
      int* mask2d, int LMAX,int Lrange2,int MMAX,int Mrange2,int Nrange, vector<int>& lindex,
      vector<int>& mindex, int FDIVX, int FDIVY,double NA, int MX, int MY, double dx,
      double dy, double lambda, int NABS, int NML, int lsmaxX, int lsmaxY,double k,
      double kx0, double ky0, vector< complex<double> >& eabs, vector< double >& dabs,
      vector< complex<double> >& cexpX, vector< complex<double> >& cexpY)
{
 double pi=atan(1.)*4.;
 complex<double>zi(0.,1.);
 complex<double> nmo(0.9238, 0.006435);
 complex<double> nsi(0.999, 0.001826);
 complex<double> nru(0.8863, 0.01706);
 complex<double> nmosi2(0.9693, 0.004333);
 complex<double> nrusi(0.9099, 0.01547);
 complex<double> nsio2(0.978, 0.01083);
 double dmo(2.052), dmosi(1.661), dsi(2.283), dsimo(1.045), dsiru(0.8), dru(2.5);
 complex<double> emo, esi, eru, emosi2, erusi, esio2;
 emo = nmo*nmo;
 esi = nsi*nsi;
 eru = nru*nru;
 emosi2 = nmosi2*nmosi2;
 erusi = nrusi*nrusi;
 esio2 = nsio2*nsio2;

 vector<Eigen::MatrixXcd> epsN(NABS, Eigen::MatrixXcd(Lrange2, Mrange2)),
 etaN(NABS, Eigen::MatrixXcd(Lrange2, Mrange2));
 vector<Eigen::MatrixXcd> zetaN(NABS, Eigen::MatrixXcd(Lrange2, Mrange2)),
 sigmaN(NABS, Eigen::MatrixXcd(Lrange2, Mrange2));
 for (int n = 0; n < NABS; n++)
 {
  Eigen::MatrixXcd eps(Lrange2,Mrange2), eta(Lrange2, Mrange2), zeta(Lrange2, Mrange2),
   sigma(Lrange2, Mrange2), leps(Lrange2, Mrange2);
  complex<double> ampta, ampvc;
  ampta=eabs[n];
  ampvc=1.;
  maskamp(FDIVX,FDIVY,NDIVX,NDIVY,mask2d, ampta, ampvc, eps,Lrange2,LMAX, Mrange2,MMAX,
   cexpX,cexpY);
  ampta=1./eabs[n];     
  maskamp(FDIVX,FDIVY,NDIVX,NDIVY,mask2d, ampta, ampvc,sigma,Lrange2,LMAX, Mrange2,MMAX,
   cexpX,cexpY);
  ampta=log(eabs[n]); 
  ampvc=0.;    
  maskamp(FDIVX,FDIVY,NDIVX,NDIVY,mask2d, ampta, ampvc, leps,Lrange2,LMAX, Mrange2,MMAX,
   cexpX,cexpY);
  epsN[n] = eps;
  sigmaN[n] = sigma;
  for (int i = 0; i < Lrange2; i++)
  {
   complex<double> zetal;
   zetal = zi*2.*pi*double(i - 2 * LMAX) / dx;
   for (int j = 0; j < Mrange2; j++)
   {
    complex<double> zetam;
    zetam = zi*2.*pi*double(j - 2 * MMAX) / dy;
    eta(i, j) = zetal*leps(i, j);
    zeta(i, j) = zetam*leps(i, j);
   }
  }
  etaN[n] = eta;
  zetaN[n] = zeta;
 }
 Eigen::MatrixXcd U0(Nrange, Nrange);
 Eigen::MatrixXcd U1U(Nrange, Nrange), U1B(Nrange, Nrange);
 Eigen::MatrixXcd B1(Nrange, Nrange);
 vector< complex<double> > al1(Nrange);
 vector<Eigen::VectorXcd> br1(Nrange, Eigen::VectorXcd(Nrange));
 vector<double> kxplus(Nrange), kyplus(Nrange), kxy2(Nrange);
 vector< complex<double> > klm(Nrange);

 for (int i = 0; i < Nrange; i++)
 {
  kxplus[i] = kx0 + 2 * pi*lindex[i] / dx;
  kyplus[i] = ky0 + 2 * pi*mindex[i] / dy;

  kxy2[i] = pow(kxplus[i], 2) + pow(kyplus[i], 2);
  klm[i] = sqrt(k*k - kxy2[i]);
 }
  Eigen::SparseMatrix<complex< double >> URUU(Nrange, Nrange), URUB(Nrange, Nrange);
  multilayerS(polar,Nrange,NML,k,kxplus,kyplus,kxy2,esio2,emo,esi,emosi2,eru,erusi,dmo,
   dmosi,dsi,dsimo,dru,dsiru,URUU,URUB);
  vector< complex<double> > alru(Nrange);
  Eigen::SparseMatrix<complex< double >> brru(Nrange, Nrange), Bru(Nrange, Nrange);
  typedef Eigen::Triplet<complex< double >> Tr;
  vector<Tr> dataBru(Nrange),databrru(Nrange);
#pragma omp parallel for
  for (int i = 0; i < Nrange; i++)
  {
   alru[i] = sqrt(k*k*eru - kxy2[i]);
   databrru[i]=Tr(i,i,1.);
   if(polar=='X')
     dataBru[i]=Tr(i, i, zi*k - zi / k / eru*pow(kxplus[i], 2));
   else if(polar=='Y')
     dataBru[i]=Tr(i, i, zi*k - zi / k / eru*pow(kyplus[i], 2));
  }
  brru.setFromTriplets(databrru.begin(),databrru.end());  
  Bru.setFromTriplets(dataBru.begin(),dataBru.end());  
/*
  vector<Eigen::VectorXcd> brru(Nrange, Eigen::VectorXcd(Nrange));
  Eigen::MatrixXcd Bru(Nrange, Nrange);
  Bru.setZero();
#pragma omp parallel for
  for (int i = 0; i < Nrange; i++)
  {
   alru[i] = sqrt(k*k*eru - kxy2[i]);
     if(polar=='X')
     Bru(i, i) = zi*k - zi / k / eru*pow(kxplus[i], 2);
   else if(polar=='Y')
     Bru(i, i) = zi*k - zi / k / eru*pow(kyplus[i], 2);
   brru[i].setZero();
   for (int j = 0; j < Nrange; j++)
   {
    if (j == i)
    {
     brru[i](j) = 1.;
    }
  }
 }
*/
 double dabs1;
 Eigen::MatrixXcd eps1(Lrange2,Mrange2), eta1(Lrange2,Mrange2), zeta1(Lrange2,Mrange2), sigma1(Lrange2,Mrange2);
 int n = NABS - 1;
 dabs1 = dabs[n];
 eps1 = epsN[n];
 eta1 = etaN[n];
 zeta1 = zetaN[n];
 sigma1 = sigmaN[n];
 absorberS0(polar,LMAX,MMAX,Nrange,lindex,mindex,k,kxplus,kyplus,kxy2,eps1,eta1,zeta1,
   sigma1,dabs1,alru,brru,Bru,URUU,URUB,al1,br1,B1,U1U,U1B);
 for (n = NABS - 2; n >= 0; n--)
 {
  dabs1 = dabs[n];
  eps1 = epsN[n];
  eta1 = etaN[n];
  zeta1 = zetaN[n];
  Eigen::MatrixXcd U2U, U2B,B2;
  vector< complex<double> > al2;
  vector<Eigen::VectorXcd> br2;
  U2U = U1U;
  U2B = U1B;
  B2=B1;
  al2 = al1;
  br2 = br1;
 absorberS(polar,LMAX,MMAX,Nrange,lindex,mindex,k,kxplus,kyplus,kxy2,eps1,eta1,zeta1,
   sigma1,dabs1,al2,br2,B2,U2U,U2B,al1,br1,B1,U1U,U1B);
 }

{
 Eigen::MatrixXcd T0L(Nrange, Nrange), T0R(Nrange, Nrange);
  #pragma omp parallel for
 for (int i = 0; i < Nrange; i++)
 {
  for (int j = 0; j < Nrange; j++)
  {
   T0L(i,j) = klm[i]*br1[j](i)+al1[j]*br1[j](i);
   T0R(i,j) = klm[i]*br1[j](i)-al1[j]*br1[j](i);
  }
 }

// U0 = T0L*U1U + T0R*U1B;
  matproduct(Nrange,T0L,U1U,T0L);
  matproduct(Nrange,T0R,U1B,T0R);
  U0=T0L+T0R;
 }
 {
  Eigen::MatrixXcd U0I(Nrange,Nrange);
  matinv(Nrange, U0,U0I);
// U1U=U1U*U0I;
// U1B=U1B*U0I;
 U1U=U1U-U1B;
  matproduct(Nrange,U1U,U0I,U1U);
 }
 Eigen::MatrixXcd FG(Nrange, Nrange);
  #pragma omp parallel for
 for (int i = 0; i < Nrange; i++)
 {
   for (int n = 0; n < Nrange; n++)
   {
    FG(i,n)=1./klm[i]*al1[n]*br1[n](i);
   }
 }

// FG=FG*(U1U-U1B);
  matproduct(Nrange,FG,U1U,FG);

//#pragma omp parallel for
 for (int ls = -lsmaxX; ls<=lsmaxX; ls++)
 {
  for (int ms = -lsmaxY; ms<=lsmaxY; ms++)
 {
 if((pow(ls*MX/dx,2)+pow(ms*MY/dy,2))<=pow(NA/lambda,2))
  {
 double kx, ky,kz;
 kx = kx0+ls*2.*pi/dx;
 ky = ky0+ms*2.*pi/dy;
 kz = sqrt(k*k-kx*kx-ky*ky);

 complex<double> Ax0p;
    Ax0p = 1.;
//    Ax0p = 1./sqrt(k*k-kx*kx);

 Eigen::VectorXcd AS(Nrange);
 AS.setZero();
 for (int i = 0; i < Nrange; i++)
  {
   if ((lindex[i] == ls) && (mindex[i] == ms))
   {
    AS(i) = 2.*kz*Ax0p;
   }
  }

/*
 Eigen::VectorXcd A1(Nrange), A1p(Nrange);
 A1 = U1U*AS;
 A1p = U1B*AS;
 Eigen::VectorXcd A1mp(Nrange);
// A1mp = A1+A1p;
 A1mp = A1-A1p;
 Eigen::VectorXcd FGA(Nrange);
 FGA=FG*A1mp;
*/
 Eigen::VectorXcd FGA(Nrange);
 FGA=FG*AS;

/*
 for (int i = 0; i < Nrange; i++)
 {
  int i2 = i + Nrange;
  complex <double> Asumx (0., 0.);
  if ((lindex[i] == ls) && (mindex[i] == ms))
  {
//   Asumx=-Ax0p;
    Asumx=Ax0p;
  }
//     Asumx+=FGA(i);
   Asumx-=FGA(i);

   Ax[ls+lsmaxX][ms+lsmaxY](i)=Asumx;
  }
*/

#pragma omp parallel for
 for (int i = 0; i < Nrange; i++)
 {
   Ax[ls+lsmaxX][ms+lsmaxY](i)=-FGA(i);
  }

 for (int i = 0; i < Nrange; i++)
 {
  if ((lindex[i] == ls) && (mindex[i] == ms))
  {
   Ax[ls+lsmaxX][ms+lsmaxY](i)+=Ax0p;
  }
 }

 }
 }
 }
}

void matinv(int Nrange, Eigen::MatrixXcd& A,Eigen::MatrixXcd& Ainv)
{
 if(Nrange>5000)
 {
 complex<double>zi(0.,1.);
 magmaDoubleComplex *mA, *dwork;
 magma_int_t N,n2,*ipiv,info,nb,lwork;
 N=Nrange;
 n2=N*N;
 nb = magma_get_zgetri_nb(N);
 lwork = N*nb;
 magma_zmalloc_pinned( &mA,    n2 );
 magma_imalloc_pinned( &ipiv,    N );
 magma_zmalloc( &dwork,  lwork );
#pragma omp parallel for
 for(int i=0;i<Nrange;i++)
 for(int j=0;j<Nrange;j++)
    mA[i+j*Nrange]=MAGMA_Z_MAKE(real(A(i,j)),imag(A(i,j)));

  magmaDoubleComplex *dA;
  info=magma_zmalloc( &dA,    n2 );
  if(info!=0)    cout<<"zmalloc "<<magma_strerror(info)<<endl;
  magma_queue_t queue;
  magma_queue_create(0, &queue);
  magma_zsetmatrix(N,N,mA,N,dA,N,queue);
  magma_zgetrf_gpu(N, N, dA, N,  ipiv, &info );
   if(info!=0)     cout<<"zgetrf "<<info<<endl;
  magma_zgetri_gpu(N,dA, N,  ipiv, dwork,lwork,&info );
  if(info!=0)    cout<<"zgetri "<<info<<endl;
  magma_zgetmatrix(N,N,dA,N,mA,N,queue);

  #pragma omp parallel for
  for(int i=0;i<Nrange;i++)
  for(int j=0;j<Nrange;j++)
   Ainv(i,j)=MAGMA_Z_REAL(mA[j*N+i])+zi*MAGMA_Z_IMAG(mA[j*N+i]);
  magma_free(dA);
  magma_free_pinned(mA);
  magma_free_pinned(ipiv);
  magma_free(dwork);
  magma_queue_destroy(queue);
 }
 else
 {
 Ainv.setIdentity();
 Ainv=A.colPivHouseholderQr().solve(Ainv);
// Ainv=A.partialPivLu().solve(Ainv);
 }
}

void matproduct(int Nrange, Eigen::MatrixXcd& A,Eigen::MatrixXcd& B,Eigen::MatrixXcd& C)
{
 if(Nrange>5000)
 {
 complex<double>zi(0.,1.);
 magma_trans_t transA=MagmaNoTrans;
 magma_trans_t transB=MagmaNoTrans;
 magmaDoubleComplex *mA,*mB, alpha,beta;
 magma_int_t N,n2;
 N=Nrange;
 n2=N*N;
 magma_zmalloc_pinned( &mA,    n2 );
 magma_zmalloc_pinned( &mB,    n2 );
  #pragma omp parallel for
 for(int i=0;i<Nrange;i++)
 for(int j=0;j<Nrange;j++)
 {
  mA[i+j*Nrange]=MAGMA_Z_MAKE(real(A(i,j)),imag(A(i,j)));
  mB[i+j*Nrange]=MAGMA_Z_MAKE(real(B(i,j)),imag(B(i,j)));
 }
 magma_int_t info;
 magmaDoubleComplex *dA,*dB, *dC;
 info=magma_zmalloc( &dA,    n2 );
 if(info!=0)   cout<<"zmalloc "<<magma_strerror(info)<<endl;
 info=magma_zmalloc( &dB,    n2 );
 if(info!=0)   cout<<"zmalloc "<<magma_strerror(info)<<endl;
 info=magma_zmalloc( &dC,    n2 );
 if(info!=0)   cout<<"zmalloc "<<magma_strerror(info)<<endl;
 magma_queue_t queue;
 magma_queue_create(0, &queue);
 magma_zsetmatrix(N,N,mA,N,dA,N,queue);
 magma_zsetmatrix(N,N,mB,N,dB,N,queue);
 alpha=MAGMA_Z_MAKE(1.,0.);
 beta=MAGMA_Z_MAKE(0.,0.);
 magmablas_zgemm(transA,transB,N,N,N,alpha,dA,N,dB,N,beta,dC,N,queue);
 magma_zgetmatrix(N,N,dC,N,mA,N,queue);
#pragma omp parallel for
  for(int i=0;i<Nrange;i++)
  for(int j=0;j<Nrange;j++)
   C(i,j)=MAGMA_Z_REAL(mA[j*N+i])+zi*MAGMA_Z_IMAG(mA[j*N+i]);
  magma_free(dA);
  magma_free(dB);
  magma_free(dC);
  magma_free_pinned(mA);
  magma_free_pinned(mB);
  magma_queue_destroy(queue);
 }
else
 {
  C=A*B;
 }
}

complex<double> fourier(int l,vector<complex<double> >& f,vector< complex<double> >& cexp,int FDIV)
{
  double div;
  div=FDIV;
 complex<double> sum;
 sum=0.;
 for (int i=0; i<FDIV; i++)
 {
   int j;
   if(l>=0) j=(i*l)%FDIV;
   else j=FDIV-((-i*l)%FDIV);
   sum=sum+f[i]*cexp[j];
  }    
  return sum/div;
}

void exponential(vector< complex<double> >& cexp,double pi,int FDIV)
{
 complex<double>zi(0.,1.);
 double div;
 div=FDIV;
 for (int i=0; i<=FDIV; i++)
 {
   complex<double> arg;
   arg=-2.*pi*zi*double(i)/div;
   cexp[i]=exp(arg);
  }    
}

