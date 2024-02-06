# Rovereye

A first attempt at a custom YOLOv8.

It's Rust because I can't stand troubleshooting Python exceptions. We'll convert it later (unless...? 😳)

## Setup

First, you'll need Rust. Follow the instructions [on this site](https://rustup.rs/) - it'll only take a minute or so to get everything running!

Afterward, go ahead and clone the repo. You can now download the model we'll be using. Stick it in the `pretrained/` folder.

Now, let's focus on platform-specific setup for OpenCV and YOLO...

### Linux

Please consider using Linux if possible - macOS has a lot of trouble linking, both with static and dynamic linking. On the other hand, Windows isn't widely supported in terms of acceleration or even general usage.

A VM will suffice if you don't mind some waiting. 😄

- Debian/Ubuntu/etc.: `sudo apt update && sudo apt upgrade -y && sudo apt install -y build-essential cmake opencv-contrib libopencv-contrib-dev libgtk2.0-dev pkg-config libboost-all-dev ffmpeg libonnx-dev libonnx1 python3`
- Fedora/RHEL/etc.: `sudo dnf update -y && sudo dnf groupinstall "Development Tools" -y && sudo dnf install -y opencv-contrib opencv-devel boost-devel ffmpeg onnxruntime onnx-devel onnxruntime-devel python3 protobuf-devel`
- Others: grab C/C++ development tools, `boost`, OpenCV, Onnx Runtime, `ffmpeg`, and a relatively modern version of Python (~3.10)

### macOS

You'll likely need to manipulate your environment variables to include a modern Clang++ version from `brew`. As such, your linker will get mad, and you'll have to fix it, too. Please try to use Linux if possible.

- If Linux isn't feasible: `brew install onnxruntime opencv ffmpeg python3 gtk4 gdk-pixbuf protobuf`

If you get an error complaining about OpenCV not having `protobuf` support compiled in, then the gem still isn't updated, and you'll need to compile OpenCV from source. (this happened in Python, too 😖)

### Windows

TODO???

There are numerous compatibility issues, so I suggest running a VM through VirtualBox or WSL.

### Model

When you download a model to train off of, make sure you convert it to ONNX using [the incredible script](https://crates.io/crates/od_opencv#user-content-prerequisites) that the Rust guy made.

Create a virtual environment and download the dependencies (`ultralytics` `onnx`). Then, you can convert it correctly!
