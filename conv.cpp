#include<iostream>
using namespace std;
#include "cnpy.h"

int HEIGHT=64;
int WIDTH=64;
int IN=64;
int OUT=128;
int OFFSET=2;
int Kw=3;
int Kh=3;
int STRIDE=2;
int OUT_H=((HEIGHT-Kh+OFFSET)/STRIDE)+1;
int OUT_W=((WIDTH-Kw+OFFSET)/STRIDE)+1;

void convv(float* input,float* weight,float* bias)
{
    
   float* output=(float *)malloc(OUT_H*OUT_W*OUT*sizeof(float));
   for(int k=0;k<OUT;k++){
      for(int i=0;i<OUT_H;i++){
        for(int j=0;j<OUT_W;j++){

                float sum=0;
                int i_stride=i*STRIDE;
                int j_stride=j*STRIDE;
                for(int o=0;o<IN;o++){
                  for(int m=0;m<Kh;m++){
                    for(int n=0;n<Kw;n++){
                        
                            int i_index=o*(HEIGHT+OFFSET)*(WIDTH+OFFSET) + (m+i_stride)*(WIDTH+OFFSET) + (j_stride+n);
                            int w_index=k*IN*Kh*Kw + o*Kh*Kw + m*Kw +n;
                            sum+=weight[w_index]*input[i_index];
                        }
                    }
                }
                output[k*OUT_H*OUT_W +i*OUT_W +j]=sum+bias[k];
            
        }
      }
    }
    std::vector<float> data(OUT*OUT_H*OUT_W);
    for(int i=0;i<OUT*OUT_H*OUT_W;i++){
      data[i]=output[i];
    }
    for(int i=0;i<10;i++){
      std::cout<<output[i]<<" ";
      
    }
    free(output);
    cnpy::npy_save("out.npy",&data[0],{1,static_cast<long unsigned>(OUT),static_cast<long unsigned>(OUT_H),static_cast<long unsigned>(OUT_W)},"w");
    
    
}
int main()
{

  float *input = (float *)std::malloc((HEIGHT+OFFSET) * (WIDTH+OFFSET) * IN * sizeof(float));
  float *weight = (float *)std::malloc(Kh*Kw*IN*OUT * sizeof(float));
  float *bias = (float *)std::malloc(OUT * sizeof(float));
  int input_size=(HEIGHT+OFFSET) * (WIDTH+OFFSET)* IN;
  int weight_size=Kh*Kw*IN*OUT;
  
  int idx = 1;
  for(int k=0; k<IN; k++){
      for(int i=0; i<HEIGHT+OFFSET; i++){
          for(int j=0; j<WIDTH+OFFSET; j++){
              if(i==0 || j==0 || i==HEIGHT+OFFSET-1 || j==WIDTH+OFFSET-1){
                  input[k*(HEIGHT+OFFSET)*(WIDTH+OFFSET) + i*(WIDTH+OFFSET) + j] = 0;
              } else {
                  input[k*(HEIGHT+OFFSET)*(WIDTH+OFFSET) + i*(WIDTH+OFFSET) + j] = idx++;
              }
          }
      }
  }
  for (int i = 0; i < weight_size; i++) {
    weight[i] = i+1;
  }
  for (int i = 0; i < OUT; i++) {
    bias[i]=0;
  }
  convv(input,weight,bias);
  free(input);
  free(weight);
  free(bias);

} 
