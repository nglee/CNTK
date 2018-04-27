## Project Description
This is a forked repository, explaining how to build CNTK 2.5 on NVIDIA's Jetson TX1. The original CNTK repository from Microsoft is [here](https://github.com/Microsoft/CNTK).

## How to Build CNTK 2.5 on TX1
It is assumed that JetPack 3.2 is installed on TX1. It is also recommended to [install additional storage device on TX1](http://www.jetsonhacks.com/2017/01/28/install-samsung-ssd-on-nvidia-jetson-tx1/) and [setup swap space](http://www.jetsonhacks.com/2016/12/21/jetson-tx1-swap-file-and-development-preparation/) to speed up build process.

Firstly, we will start by installing prerequisites in the following list. Most of the guides for installing these prerequisites are from the [Microsoft CNTK Document "Setup CNTK on Linux"](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux).

* OpenBLAS 0.2.20
* libzip 1.1.2
* Boost 1.60.0
* CUB 1.7.4
* Protobuf 3.1.0
* OpenCV 3.1.0

After these prerequisites are setup, we will create some symbolic links, and then finally build CNTK 2.5 itself.

### OpenBLAS 0.2.20
```
$ git clone https://github.com/xianyi/OpenBLAS.git
$ cd OpenBLAS
$ git checkout v0.2.20
$ sudo apt install gfortran
$ make -j4
$ sudo make install PREFIX=/usr/local/OpenBLAS
```
### libzip 1.1.2
```
$ wget http://nih.at/libzip/libzip-1.1.2.tar.gz
$ tar xzf ./libzip-1.1.2.tar.gz
$ cd libzip-1.1.2
$ ./configure
$ make -j4
$ sudo make install
```
### Boost 1.60.0
```
$ wget -q -O - https://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz/download | tar -xzf -
$ cd boost_1_60_0
$ ./bootstrap.sh --prefix=/usr/local/boost-1.60.0
$ sudo apt install libbz2-dev python-dev
$ sudo ./b2 -d0 -j4 install
```
### CUB 1.7.4
```
$ wget https://github.com/NVlabs/cub/archive/1.7.4.zip
$ unzip ./1.7.4.zip
$ sudo cp -r cub-1.7.4 /usr/local
```
### Protobuf 3.1.0
```
$ wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz
$ tar xzf v3.1.0.tar.gz
$ cd protobuf-3.1.0
$ sudo apt install curl
$ ./autogen.sh
$ ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/protobuf-3.1.0
$ make -j4
$ sudo make install
```
### OpenCV 3.1.0
```
$ sudo apt install cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
$ wget https://github.com/Itseez/opencv/archive/3.1.0.zip
$ unzip 3.1.0.zip
$ cd opencv-3.1.0
$ mkdir release
$ cd release
$ cmake -D WITH_CUDA=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.1.0 -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF ..
$ make all
$ sudo make install
```
### Create Some Symbolic Links
```
# create symbolic links for CUDNN
$ sudo mkdir /usr/local/cudnn-7.0/cuda/include -p
$ sudo ln -s /usr/lib/aarch64-linux-gnu /usr/local/cudnn-7.0/lib
$ sudo ln -s /usr/include/aarch64-linux-gnu/cudnn_v7.h /usr/local/cudnn-7.0/cuda/include/cudnn.h

# create symbolic link for NVML
$ sudo ln -s /usr/local/cuda-9.0/targets/aarch64-linux/include/nvml.h /usr/local/include

# create symbolic link for Open MPI C++ wrapper compiler
$ sudo mkdir /usr/lib/openmpi/bin
$ sudo ln -s /usr/bin/mpic++ /usr/lib/openmpi/bin/mpic++
```
### Build CNTK 2.5
```
$ git clone https://github.com/nglee/CNTK.git
$ cd CNTK
$ git checkout v2.5_tx1
$ mkdir build/release -p
$ cd build/release
$ ../../configure --asgd=no \
                  --cuda=yes \
                  --with-openblas=/usr/local/OpenBLAS \
                  --with-boost=/usr/local/boost-1.60.0 \
                  --with-cudnn=/usr/local/cudnn-7.0 \
                  --with-protobuf=/usr/local/protobuf-3.1.0 \
                  --with-mpi=/usr/lib/openmpi \
                  --with-gdk-include=/usr/local/include \
                  --with-gdk-nvml-lib=/usr/local/cuda-9.0/targets/aarch64-linux/lib/stubs
$ make -C ../../ \
       BUILD_TOP=$PWD \
       SSE_FLAGS='' \
       GENCODE_FLAGS='-gencode arch=compute_53,code=\"sm_53,compute_53\"' \
       all \
       -j 4
```
After a successful build, we'll find CNTK 2.5 libraries and binaries placed under `lib` and `bin` directoy. Those directories are created at `[CNTK_SOURCE_BASE]/build/release`.

## How to Test the Build
To ensure that CNTK is working properly, we can follow the test steps described [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#quick-test-of-cntk-build-functionality). However, TX1 needs some extra steps regarding NVML(NVIDIA Management Library). As mentioned [here](https://devtalk.nvidia.com/default/topic/999740/jetson-tx1/-libnvidia-ml-so-1-cannot-open-shared-object-file-no-such-file-or-directory-with-cross-compiled-elf/post/5108791/#5108791), NVML does not support TX1, and it just provides a stub library. Since the binary called `cntk` that is used for the test needs NVML, we will setup the stub library for `cntk`.
```
$ export LD_LIBRARY_PATH=/usr/local/cuda-9.0/targets/aarch64-linux/lib/stubs:$LD_LIBRARY_PATH
$ sudo ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/stubs/libnvidia-ml.so /usr/local/cuda-9.0/targets/aarch64-linux/lib/stubs/libnvidia-ml.so.1
```
After setting up the stub library, let's try CNTK with CPU.
```
$ export PATH=$HOME/Repos/cntk/build/release/bin:$PATH
$ cd [CNTK_SOURCE_BASE]/Tutorials/HelloWorld-LogisticRegression
$ cntk configFile=lr_bs.cntk makeMode=false
```
To try CNTK with GPU:
```
$ cntk configFile=lr_bs.cntk makeMode=false deviceId=auto
```

## Contribute
Contributions are always welcome. You can just start a pull request.

## Credits
[Microsoft CNTK Document "Setup CNTK on Linux"](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux)  
[Why do we need `gfortran` when building OpenBLAS](https://github.com/Microsoft/CNTK/issues/1424)  
[NVML does not support Tegra platform](https://devtalk.nvidia.com/default/topic/999740/jetson-tx1/-libnvidia-ml-so-1-cannot-open-shared-object-file-no-such-file-or-directory-with-cross-compiled-elf/post/5108791/#5108791)

## License
The original CNTK library from Microsoft is [released under MIT license](https://github.com/Microsoft/CNTK/blob/v2.5/LICENSE.md). So is this port.

