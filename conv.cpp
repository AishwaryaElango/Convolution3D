#include <iostream>
using namespace std;
#include "cnpy.h"

int HEIGHT = 64;
int WIDTH = 64;
int IN = 64;
int OUT = 128;
int OFFSET = 2;
int Kw = 3;
int Kh = 3;
int STRIDE = 2;
int OUT_H = ((HEIGHT - Kh + OFFSET) / STRIDE) + 1;
int OUT_W = ((WIDTH - Kw + OFFSET) / STRIDE) + 1;

void conv_nhwc(float *input, float *weight, float *bias) {
  float *output = (float *)malloc(OUT_H * OUT_W * OUT * sizeof(float));
  for (int i = 0; i < OUT_H; i++) {
    for (int j = 0; j < OUT_W; j++) {
      for (int k = 0; k < OUT; k++) {
        float sum = 0;
        int i_stride = i * STRIDE;
        int j_stride = j * STRIDE;
        for (int m = 0; m < Kh; m++) {
          for (int n = 0; n < Kw; n++) {
            for (int o = 0; o < IN; o++) {
              int i_index = (i_stride + m) * (WIDTH + OFFSET) * IN +
                            (j_stride + n) * IN + o;
              int w_index = (m * Kw * IN * OUT) + n * IN * OUT + o * OUT + k;
              sum += input[i_index] * weight[w_index];
            }
          }
        }
        output[i * OUT_W * OUT + j * OUT + k] = sum + bias[k];
      }
    }
  }
  std::cout<<"\n======================================================================\n";
  std::cout<<"Cpp Outputs ";
  std::cout<<"\n======================================================================\n";
  for (int i = 0; i < 10; i++) {
    std::cout << output[i] << " ";
  }
  std::cout<<"\n";
  std::vector<float> data(OUT * OUT_H * OUT_W);
  for (int i = 0; i < OUT * OUT_H * OUT_W; i++) {
    data[i] = output[i];
  }
  free(output);
  cnpy::npy_save(
      "out2.npy", &data[0],
      {1, static_cast<long unsigned>(OUT_H), static_cast<long unsigned>(OUT_W),
       static_cast<long unsigned>(OUT)},
      "w");
}

void conv_nchw(float *input, float *weight, float *bias) {
  float *output = (float *)malloc(OUT_H * OUT_W * OUT * sizeof(float));
  for (int k = 0; k < OUT; k++) {
    for (int i = 0; i < OUT_H; i++) {
      for (int j = 0; j < OUT_W; j++) {
        float sum = 0;
        int i_stride = i * STRIDE;
        int j_stride = j * STRIDE;
        for (int o = 0; o < IN; o++) {
          for (int m = 0; m < Kh; m++) {
            for (int n = 0; n < Kw; n++) {
              int i_index = o * (HEIGHT + OFFSET) * (WIDTH + OFFSET) +
                            (m + i_stride) * (WIDTH + OFFSET) + (j_stride + n);
              int w_index = k * IN * Kh * Kw + o * Kh * Kw + m * Kw + n;
              sum += weight[w_index] * input[i_index];
            }
          }
        }
        output[k * OUT_H * OUT_W + i * OUT_W + j] = sum + bias[k];
      }
    }
  }
  std::vector<float> data(OUT * OUT_H * OUT_W);
  for (int i = 0; i < OUT * OUT_H * OUT_W; i++) {
    data[i] = output[i];
  }
  std::cout<<"======================================================================\n";
  std::cout<<"Cpp Outputs ";
  std::cout<<"\n======================================================================\n";
  for (int i = 0; i < 10; i++) {
    std::cout << output[i] << " ";
  }
  std::cout<<"\n";
  free(output);
  cnpy::npy_save(
      "out.npy", &data[0],
      {1, static_cast<long unsigned>(OUT), static_cast<long unsigned>(OUT_H),
       static_cast<long unsigned>(OUT_W)},
      "w");
}
int main() {
  float *input_nchw = (float *)std::malloc((HEIGHT + OFFSET) * (WIDTH + OFFSET) *
                                      IN * sizeof(float));
  float *input_nhwc = (float *)std::malloc((HEIGHT + OFFSET) * (WIDTH + OFFSET) *
                                      IN * sizeof(float));                                  
  float *weight = (float *)std::malloc(Kh * Kw * IN * OUT * sizeof(float));
  float *bias = (float *)std::malloc(OUT * sizeof(float));
  int input_size = (HEIGHT + OFFSET) * (WIDTH + OFFSET) * IN;
  int weight_size = Kh * Kw * IN * OUT;

  int idx = 1;
  for (int k = 0; k < IN; k++) {
    for (int i = 0; i < HEIGHT + OFFSET; i++) {
      for (int j = 0; j < WIDTH + OFFSET; j++) {
        if (i == 0 || j == 0 || i == HEIGHT + OFFSET - 1 ||
            j == WIDTH + OFFSET - 1) {
          input_nchw[k * (HEIGHT + OFFSET) * (WIDTH + OFFSET) +
                i * (WIDTH + OFFSET) + j] = 0;
        } else {
          input_nchw[k * (HEIGHT + OFFSET) * (WIDTH + OFFSET) +
                i * (WIDTH + OFFSET) + j] = idx++;
        }
      }
    }
  }

  // Uncomment to use NHWC Format

  idx = 1;
  for (int i = 0; i < HEIGHT + OFFSET; i++) {
    for (int j = 0; j < WIDTH + OFFSET; j++) {
      for (int k = 0; k < IN; k++) {
        if (i == 0 || j == 0 || i == HEIGHT + OFFSET - 1 ||
            j == WIDTH + OFFSET - 1) {
          input_nhwc[i * (WIDTH + OFFSET) * IN + j * IN + k] = 0;
        } else {
          input_nhwc[i * (WIDTH + OFFSET) * IN + j * IN + k] = idx++;
        }
      }
    }
  }

  for (int i = 0; i < weight_size; i++) {
    weight[i] = i + 1;
  }
  for (int i = 0; i < OUT; i++) {
    bias[i] = 0;
  }
  conv_nchw(input_nchw, weight, bias);
  conv_nhwc(input_nhwc,weight,bias);
  free(input_nchw);
  free(input_nhwc);
  free(weight);
  free(bias);
}
