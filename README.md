# Rovereye

A first attempt at a custom YOLOv8.

It's Rust because I can't stand troubleshooting Python exceptions. We'll convert it later (unless...? 😳)

<img src="images/flowers_yolov8_n.jpg" height="320" alt="A sample output image of YOLOv8 with bounding box, classification, and confidence"></img>

## Setup

First, you'll need Rust. Follow the instructions [on this site](https://rustup.rs/) - it'll only take a minute or so to get everything running!

Afterward, go ahead and clone the repo. You can now download the model we'll be using. Stick it in the `pretrained/` folder.

Now, let's focus on platform-specific setup for OpenCV and YOLO...

### Linux

Please consider using Linux if possible - macOS has a lot of trouble linking, both with static and dynamic linking. On the other hand, Windows isn't widely supported in terms of acceleration or even general usage.

A VM will suffice if you don't mind some waiting. 😄

Anyway, run one of these commands on your machine according to its distribution:

- Debian/Ubuntu/etc.: `sudo apt update && sudo apt upgrade -y && sudo apt install -y build-essential cmake llvm libclang-dev clang libopencv-dev libgtk2.0-dev pkg-config libboost-all-dev ffmpeg libonnx-dev libonnx1 python3`
- Fedora/RHEL/etc.: `sudo dnf update -y && sudo dnf groupinstall "Development Tools" -y && sudo dnf install -y clang-devel libxcrypt-compat gcc cmake python3-devel llvm-devel opencv-contrib opencv-devel boost-devel ffmpeg onnxruntime onnx-devel onnxruntime-devel python3 protobuf-devel gstreamer1-plugins-base gstreamer1-plugins-good "gstreamer1-plugins-bad-*" "gstreamer1-plugins-ugly-*" gstreamer1-plugin-openh264 gstreamer1-plugin-libav ffmpeg`
- Others: grab C/C++ development tools, `boost`, OpenCV, `protobuf`, Onnx Runtime, `ffmpeg`, and a relatively modern version of Python (~3.10)

#### Compiling OpenCV with NVIDIA CUDNN Support

To allow NVIDIA cards to work with OpenCV's DNN module, you'll need to build the module with CUDA support. Let's begin!

1. Install the CUDA toolkit. You can find it [here](https://developer.nvidia.com/cuda-downloads).
    - If your distribution is supported, you'll [find it here](https://developer.download.nvidia.com/compute/cuda/repos/).
    - Otherwise, you'll have to use a Docker container. (fair warning: i haven't done this before)
    - On Fedora, there's [a nice guide on RPM Fusion](https://rpmfusion.org/Howto/CUDA#CUDA_Toolkit).
    - You'll also need to grab the `dnn` packages. NVIDIA documented [a mildly-annoying method here](https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html#rhel-9-rocky-9-and-rhel-8-rocky-8-network-installation). Use `rhel9` on Fedora.
1. Make sure you've installed the dependencies listed above.
    - The rest of this guide will focus on Fedora.
1. Let's grab some build dependencies!
    - First, [install RPM Fusion](https://docs.fedoraproject.org/en-US/quick-docs/rpmfusion-setup/).
    - Now, you can install these build dependencies: `sudo dnf install --allowerasing gcc-c++ cmake chrpath libtheora-devel libvorbis-devel libraw1394-devel libdc1394-devel jasper-devel libjpeg-devel libpng-devel libtiff-devel libGL-devel libv4l-devel OpenEXR-devel openni-devel openni-primesense tbb-devel zlib-devel pkgconfig python3-devel python3-numpy python3-setuptools pylint python3-flake8 swig ffmpeg-libs libavdevice gstreamer1-devel gstreamer1-plugins-base-devel opencl-headers libgphoto2-devel libwebp-devel tesseract-devel protobuf-devel gdal-devel glog-devel python3-beautifulsoup4 gflags-devel qt5-qtbase-devel libGLU-devel hdf5-devel openjpeg2-devel freetype-devel harfbuzz-devel vulkan-headers libvpl-devel gtk3-devel gtk4-devel`
    - Before running this command, ensure that nothing important is being removed! (the `--allowerasing` flag helps remove certain packages that conflict, but be careful!)
1. Head to a place you don't care about and `git clone https://github.com/opencv/opencv`
1. You'll also want to grab the `opencv_contrib` set of modules: `git clone https://github.com/opencv/opencv_contrib`
1. `cd opencv`
1. `mkdir build`
1. `cd build`
1. Run the following command to get ready for the build: `cmake -DCMAKE_CXX_STANDARD=17 -DCV_TRACE=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_QT=ON -DWITH_OPENGL=ON -DBUILD_TESTS=OFF -DOpenGL_GL_PREFERENCE=GLVND -DWITH_GDAL=ON -DWITH_OPENEXR=ON -DCMAKE_SKIP_RPATH=ON -DWITH_CAROTENE=OFF -DCPU_BASELINE=SSE2 -DCMAKE_BUILD_TYPE=Release -DWITH_GSTREAMER=ON -DWITH_FFMPEG=ON -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" -DCUDA_VERBOSE_BUILD=ON -DCUDA_PROPAGATE_HOST_FLAGS=OFF -DCUDA_NVCC_FLAGS="-Xcompiler -fPIC" -DOPENCV_DNN_CUDA=ON -DWITH_OPENNI=ON -DWITH_XINE=ON -DBUILD_DOCS=ON -DBUILD_EXAMPLES=ON -DBUILD_opencv_python2=OFF -DINSTALL_C_EXAMPLES=ON -DINSTALL_PYTHON_EXAMPLES=ON -DPYTHON3_EXECUTABLE="/usr/bin/python3" -DOPENCV_GENERATE_SETUPVARS=OFF -DENABLE_PYLINT=OFF -DENABLE_FLAKE8=OFF -DBUILD_PROTOBUF=OFF -DPROTOBUF_UPDATE_FILES=ON -DOPENCV_DNN_OPENCL=ON -DOPENCL_INCLUDE_DIR=/usr/include/CL -DOPENCV_DNN_OPENCL=ON -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib/modules" -DWITH_LIBV4L=ON -DWITH_OPENMP=ON -DOPENCV_CONFIG_INSTALL_PATH="/lib64/cmake/OpenCV" -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_MFX=ON -DWITH_GAPI_ONEVPL=ON -DWITH_VA=ON -DWITH_VULKAN=ON -DVULKAN_INCLUDE_DIRS="/usr/include/vulkan" ../`
    - this may not compile yet - needs FFmpeg with NVIDIA support, which requires compiling all of FFmpeg with the proprietary NVIDIA dependencies and drivers. Take a look [at this article](https://www.cyberciti.biz/faq/how-to-install-ffmpeg-with-nvidia-gpu-acceleration-on-linux/) if you feel like giving it a shot. I don't.
1. Start compiling with `make -j7`.
1. Install it to your system with `make -j7 install`.
1. Run `opencv_version` and make sure it says `4.(something)-dev` in response!

### macOS

You'll likely need to manipulate your environment variables to include a modern Clang++ version from `brew`. As such, your linker will get mad, and you'll have to fix it, too. Please try to use Linux if possible.

- If Linux isn't feasible: `brew install onnxruntime ffmpeg python3 gtk4 gdk-pixbuf protobuf`

#### Compiling OpenCV with Protobuf

If you get an error complaining about OpenCV not having `protobuf` support compiled in, then you're using the Homebrew version.

You need to compile it yourself. Here are some instructions...

1. Head to a place you don't care about and `git clone https://github.com/opencv/opencv`
1. You'll also want to grab the `opencv_contrib` set of modules: `git clone https://github.com/opencv/opencv_contrib`
1. `cd opencv`
1. `mkdir build`
1. `cd build`
1. Run this giant command: `cmake -DCMAKE_CXX_STANDARD=11 -DCMAKE_OSX_DEPLOYMENT_TARGET= -DBUILD_JASPER=OFF -DBUILD_JPEG=OFF -DBUILD_OPENEXR=OFF -DBUILD_OPENJPEG=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=OFF -DBUILD_PROTOBUF=ON -DBUILD_TBB=OFF -DBUILD_TESTS=OFF -DBUILD_TIFF=OFF -DBUILD_WEBP=OFF -DBUILD_ZLIB=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_text=ON -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib/modules" -DOPENCV_GENERATE_PKGCONFIG=ON -DPROTOBUF_UPDATE_FILES=ON -DWITH_1394=OFF -DWITH_CUDA=OFF -DWITH_EIGEN=ON -DWITH_FFMPEG=ON -DWITH_GPHOTO2=OFF -DWITH_GSTREAMER=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=ON -DWITH_OPENGL=OFF -DWITH_OPENVINO=ON-DWITH_QT=OFF -DWITH_TBB=ON-DWITH_VTK=ON -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON -DPYTHON3_EXECUTABLE="/opt/homebrew/bin/python3" -DCMAKE_BUILD_TYPE=Release ../`
1. Start compiling with `make -j7`.
1. Install it to your system with `make -j7 install`.
1. Run `opencv_version` and make sure it says `4.(something)-dev` in response!

### Windows

TODO???

There are numerous compatibility issues, so I suggest running a VM through VirtualBox or WSL.

### Model

When you download a model to train from, make sure you convert it to ONNX using [the incredible script](https://crates.io/crates/od_opencv#user-content-prerequisites) that the Rust guy made.

Create a virtual environment and download the dependencies (`ultralytics` `onnx`). Then, you can convert it correctly!

For additional instructions, see [training/README.md](training/README.md).
