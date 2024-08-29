import torch
from torch import nn
import numpy as np
import sys

INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
ONE_BYTE = 8
OFFSET = 2
PADDING = 0
STRIDE = 2

if __name__ == "__main__":
    # fill input with random numbers
    input_npy = [i +1  for i in range(HEIGHT * WIDTH * INPUT_CHANNELS)]
    input_npy = np.array(input_npy)
    input_npy = input_npy.reshape(1, INPUT_CHANNELS, HEIGHT, WIDTH)
    input_npy = input_npy.astype(np.float32)
    input_tensor = torch.from_numpy(input_npy)

    # add padding to the input
    input_tensor = torch.nn.functional.pad(
        input_tensor, (1, 1, 1, 1), mode="constant", value=0
    )

    # fill weight with random numbers
    weight_npy = [
        i + 1
        for i in range(OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)
    ]
    weight_npy = np.array(weight_npy)
    weight_npy = weight_npy.reshape(
        OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE
    )
    weight_npy = weight_npy.astype(np.float32)
    weight_tensor = torch.from_numpy(weight_npy)
    
    
    # define the convolution layer
    conv = nn.Conv2d(
        INPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=0, bias=False, stride=2
    )
    conv.weight.data = weight_tensor


    output = conv(input_tensor)
    output_nhwc=output.permute(0,2,3,1)
    print("\n======================================================================")
    print("Python outputs ")
    print("======================================================================")
    print("\n======================================================================")
    print("NCHW Output ")
    print("======================================================================")
    print(output.flatten()[0:10])
    print("\n======================================================================")
    print("NHWC Output ")
    print("======================================================================")
    print(output_nhwc.flatten()[0:10])
    np.save("py_conv_nchw_output.npy", output.detach().numpy())
    np.save("py_conv_nhwc_output.npy", output_nhwc.detach().numpy())
    print("outputs dumped to py_conv_output.npy files")
    print("\n======================================================================")
    print("Verification ")
    print("======================================================================")
