#include <iostream>
#include<fstream>
#include <vector>
#include <cstdlib>
using namespace std;

#include "./mVDF.h"

vector<uint8_t> compressBits(vector<bool>& bits) {
    vector<uint8_t> bytes;
    uint8_t byte = 0;
    int count = 0;
    for (bool bit : bits) {
        byte |= (bit << (7 - count));  
        count++;
        if (count == 8) {
            bytes.push_back(byte);
            byte = 0;
            count = 0;
        }
    }
    if (count > 0) {
        bytes.push_back(byte); 
    }
    return bytes;
}

int main (int argc,char* argv[])
{
 int ndata=1;
// srand(time(NULL));
 int NDIVX =2048;
// int NDIVX =1024;
 int NDIVY =NDIVX;
 int NDIVSQ=NDIVX*NDIVY;
 int* mask2d=new int[NDIVSQ];
 vector<bool> bits(NDIVSQ);
 ofstream ofs("mask.bin");
 ofstream ofsmask("maskimage.csv");

 for(int nsample=1;nsample<=ndata;nsample++)
 {
  maskgen(mask2d,NDIVX,NDIVY);
  for(int i=0;i<NDIVSQ;i++)
   bits[i]=static_cast<bool>(mask2d[i]);
  auto bytes= compressBits(bits);
  for (uint8_t b : bytes) ofs << b;

   if(nsample==1)
 {
   for (int j = 0; j < NDIVY; j++)
   {
    for (int i = 0; i < NDIVX; i++)
    {
      ofsmask<<mask2d[NDIVY*i +NDIVY-1-j]<<",";
    }
    ofsmask<<endl;
   }
  }
 }
 return 0;
}

