#!/bin/bash

# Compile the C++ code
g++ -o conv_3d conv.cpp -L/home/ubuntu/cnpy -lcnpy -lz --std=c++11

# Run the compiled C++ program
./conv_3d

# Run the Python script for inference
python inference_nchw.py

# Run the Python script to compare outputs
python compare.py py_conv_nchw_output.npy out.npy

python inference_nhwc.py

python compare.py py_conv_output_nhwc.npy out2.npy
