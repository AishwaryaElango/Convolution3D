import torch
from torch import nn
import numpy as np

# Define the parameters
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1  # No padding applied

if __name__ == "__main__":
    # Initialize input with sequential values for easier debugging
    input_npy = np.arange(1, HEIGHT * WIDTH * INPUT_CHANNELS + 1, dtype=np.float32)
    input_npy = input_npy.reshape(1, HEIGHT, WIDTH, INPUT_CHANNELS)  

    # Convert NHWC to NCHW format
    input_tensor = torch.from_numpy(input_npy).permute(0, 3, 1, 2)  # Now in NCHW format

    # Initialize weights with sequential values for easier debugging
    weight_npy = np.arange(1, OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE + 1, dtype=np.float32)
    weight_npy = weight_npy.reshape( KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS,OUTPUT_CHANNELS)  

    weight_tensor = torch.from_numpy(weight_npy).permute(3,2,0,1)
    
    # Define the convolution layer
    conv = nn.Conv2d(INPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, stride=STRIDE, bias=False)
    conv.weight.data = weight_tensor

    # Perform the convolution in NCHW format
    output_nchw = conv(input_tensor)

    # Convert the output from NCHW back to NHWC format (optional)
    output_nhwc = output_nchw.permute(0, 2, 3, 1)

    # Print output shape to verify
    print("Output shape (NHWC):", output_nhwc.shape)

    # Print the first few values of the output tensor
    print("Output values:", output_nhwc.flatten()[0:10])

    # Optionally save the output for comparison or further analysis
    np.save("py_conv_output_nhwc.npy", output_nhwc.detach().numpy())

