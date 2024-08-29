#include <iostream>
using namespace std;
#include "cnpy.h"

int HEIGHT = 64;
int WIDTH = 64;
int IN_CHANNEL = 64;
int OUT_CHANNEL = 128;
int OFFSET = 2;
int Kw = 3;
int Kh = 3;
int STRIDE = 2;
int OUT_H = ((HEIGHT - Kh + OFFSET) / STRIDE) + 1;
int OUT_W = ((WIDTH - Kw + OFFSET) / STRIDE) + 1;

void conv_nhwc(float *input, float *weight, float *bias) {
  float *output = (float *)malloc(OUT_H * OUT_W * OUT_CHANNEL * sizeof(float));
  for (int i = 0; i < OUT_H; i++) {
    for (int j = 0; j < OUT_W; j++) {
      for (int k = 0; k < OUT_CHANNEL; k++) {
        float sum = 0;
        int i_stride = i * STRIDE;
        int j_stride = j * STRIDE;
        for (int m = 0; m < Kh; m++) {
          for (int n = 0; n < Kw; n++) {
            for (int o = 0; o < IN_CHANNEL; o++) {
              int i_index = (i_stride + m) * (WIDTH + OFFSET) * IN_CHANNEL +
                            (j_stride + n) * IN_CHANNEL + o;
              int w_index = (m * Kw * IN_CHANNEL * OUT_CHANNEL) + n * IN_CHANNEL * OUT_CHANNEL + o * OUT_CHANNEL + k;
              sum += input[i_index] * weight[w_index];
            }
          }
        }
        output[i * OUT_W * OUT_CHANNEL + j * OUT_CHANNEL + k] = sum + bias[k];
      }
    }
  }
  std::cout << "\n============================================================="
               "=========\n";
  std::cout << "Cpp Outputs ";
  std::cout << "\n============================================================="
               "=========\n";
  for (int i = 0; i < 10; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  std::vector<float> data(OUT_CHANNEL * OUT_H * OUT_W);
  for (int i = 0; i < OUT_CHANNEL * OUT_H * OUT_W; i++) {
    data[i] = output[i];
  }
  free(output);
  cnpy::npy_save(
      "out2.npy", &data[0],
      {1, static_cast<long unsigned>(OUT_H), static_cast<long unsigned>(OUT_W),
       static_cast<long unsigned>(OUT_CHANNEL)},
      "w");
}

void conv_nchw(float *input, float *weight, float *bias) {
  float *output = (float *)malloc(OUT_H * OUT_W * OUT_CHANNEL * sizeof(float));
  for (int k = 0; k < OUT_CHANNEL; k++) {
    for (int i = 0; i < OUT_H; i++) {
      for (int j = 0; j < OUT_W; j++) {
        float sum = 0;
        int i_stride = i * STRIDE;
        int j_stride = j * STRIDE;
        for (int o = 0; o < IN_CHANNEL; o++) {
          for (int m = 0; m < Kh; m++) {
            for (int n = 0; n < Kw; n++) {
              int i_index = o * (HEIGHT + OFFSET) * (WIDTH + OFFSET) +
                            (m + i_stride) * (WIDTH + OFFSET) + (j_stride + n);
              int w_index = k * IN_CHANNEL * Kh * Kw + o * Kh * Kw + m * Kw + n;
              sum += weight[w_index] * input[i_index];
            }
          }
        }
        output[k * OUT_H * OUT_W + i * OUT_W + j] = sum + bias[k];
      }
    }
  }
  std::vector<float> data( OUT_CHANNEL * OUT_H * OUT_W);
  for (int i = 0; i < OUT_CHANNEL * OUT_H * OUT_W; i++) {
    data[i] = output[i];
  }
  std::cout << "==============================================================="
               "=======\n";
  std::cout << "Cpp Outputs ";
  std::cout << "\n============================================================="
               "=========\n";
  for (int i = 0; i < 10; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  free(output);
  cnpy::npy_save(
      "out.npy", &data[0],
      {1, static_cast<long unsigned>(OUT_CHANNEL), static_cast<long unsigned>(OUT_H),
       static_cast<long unsigned>(OUT_W)},
      "w");
}
int main() {
  float *input_nchw = (float *)std::malloc(
      (HEIGHT + OFFSET) * (WIDTH + OFFSET) * IN_CHANNEL * sizeof(float));
  float *input_nhwc = (float *)std::malloc(
      (HEIGHT + OFFSET) * (WIDTH + OFFSET) * IN_CHANNEL * sizeof(float));
  float *weight_nchw = (float *)std::malloc(Kh * Kw * IN_CHANNEL * OUT_CHANNEL * sizeof(float));
  float *weight_nhwc = (float *)std::malloc(Kh * Kw * IN_CHANNEL * OUT_CHANNEL * sizeof(float));
  float *bias = (float *)std::malloc(OUT_CHANNEL * sizeof(float));
  int input_size = (HEIGHT + OFFSET) * (WIDTH + OFFSET) * IN_CHANNEL;
  int weight_size = Kh * Kw * IN_CHANNEL * OUT_CHANNEL;

  int idx = 1;
  for (int k = 0; k < IN_CHANNEL; k++) {
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

  
  for (int k = 0; k < IN_CHANNEL; k++) {
    for (int i = 0; i < HEIGHT + OFFSET; i++) {
      for (int j = 0; j < WIDTH + OFFSET; j++) {
        int nhwc_idx = i * (WIDTH + OFFSET) * IN_CHANNEL + j * IN_CHANNEL + k;
        if (i == 0 || j == 0 || i == HEIGHT + OFFSET - 1 ||
            j == WIDTH + OFFSET - 1) {
          input_nhwc[nhwc_idx] = 0;
        } else {
          input_nhwc[nhwc_idx] =
              input_nchw[k * (HEIGHT + OFFSET) * (WIDTH + OFFSET) +
                         i * (WIDTH + OFFSET) + j];
        }
      }
    }
  }

  idx = 1;
  for (int i = 0; i < OUT_CHANNEL; i++) {
    for (int j = 0; j < IN_CHANNEL; j++) {
      for (int k = 0; k < Kh; k++) {
        for (int l = 0; l < Kw; l++) {
          weight_nchw[i * IN_CHANNEL * Kh * Kw + j * Kh * Kw + k * Kw + l] = idx++;
        }
      }
    }
  }

  
  for (int i = 0; i < OUT_CHANNEL; i++) {
    for (int j = 0; j < IN_CHANNEL; j++) {
      for (int k = 0; k < Kh; k++) {
        for (int l = 0; l < Kw; l++) {
          int nhwc_idx = k * Kw * IN_CHANNEL * OUT_CHANNEL + l * IN_CHANNEL * OUT_CHANNEL + j * OUT_CHANNEL + i;
          weight_nhwc[nhwc_idx] =
              weight_nchw[i * IN_CHANNEL * Kh * Kw + j * Kh * Kw + k * Kw + l];
        }
      }
    }
  }

  std::cout << "\n";
  
  for (int i = 0; i < OUT_CHANNEL; i++) {
    bias[i] = 0;
  }
  conv_nchw(input_nchw, weight_nchw, bias);
  conv_nhwc(input_nhwc, weight_nhwc, bias);
  free(input_nchw);
  free(input_nhwc);
  free(weight_nchw);
  free(weight_nhwc);
  free(bias);
}
