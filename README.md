# Prerequisites
- **C++ Compiler** (e.g., GCC)
- **Cnpy Library**
  
# Steps to build and install cnpy

 ```

        git clone https://github.com/rogersce/cnpy.git 
        cd cnpy
        mkdir build
        cd build
        cmake ..
        make
        sudo make install

 ```

# How to run and verify the C++ kernel 

1. Compile the kernel
    ```
        g++ -o conv_3d conv.cpp -L/path/to/install/dir -lcnpy -lz --std=c++11
    ```
2. Run the Program
    ```
        ./conv_3d
    ```
3. Run the Python inference script
    ```
        python inf.py
    ```
4. Compare the python and cpp outputs
    ```
        python compare.py py_conv_output.npy out.npy
    ```
