void randmask(int *mask1d, int NDIV, int gap);
void maskgen(int *mask2d, int NDIVX, int NDIVY)
{
  int* mask1dX=new int[NDIVX];
  int* mask1dY=new int[NDIVY];
  int cd = 56;
  int gap=80;
 int sum, space;
 space=rand()%2;
 sum=0; 
 while (sum < NDIVX-cd)
 { 
    if(space==1)
    {
       for(int i=sum;i<min(sum+cd,NDIVX);i++)  
        for(int j=0;j<NDIVY;j++)
         mask2d[NDIVX*i+j]=0;
        space=0;
      }
      else
    {  
       randmask(mask1dY,NDIVY,gap);
       for(int i=sum;i<min(sum+cd,NDIVX);i++)  
       for(int j=0;j<NDIVY;j++)
        mask2d[NDIVX*i+j]=mask1dY[j];
        space=1;
      }
    sum=sum+cd;
  }
 for(int i=sum;i<NDIVX;i++)
  {
   if(space == 0)
     for(int j=0;j<NDIVY;j++)
     mask2d[NDIVX*i+j]=0;
   else
   for(int j=0;j<NDIVY;j++)
       mask2d[NDIVX*i+j]=mask1dY[j];
  }
}

void randmask(int *mask1d, int NDIV, int gap)
{
 int sum,a, line;
 line=rand()%2;
 sum=0;    
 while (sum < NDIV-gap)
 { 
 a=gap*(rand()%5);
  if(line==1) a=a+3*gap;
 for(int i=sum;i<min(sum+a,NDIV);i++)
      {
     if(line == 1)
      mask1d[i] = 1;
      else
      mask1d[i] = 0;
     }
    if(line == 1)
     line = 0;
    else
     line = 1;
    sum = sum + a;
  }
 for(int i=sum;i<NDIV;i++)
  {
   if(line == 0)
    mask1d[i] = 1;
   else
    mask1d[i] = 0;
  }
}

