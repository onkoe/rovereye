# training

Trains the model based on a given dataset.

You can also use `export.py` to get an ONNX export. OpenCV will like that... 😄️

## Installation

1. Clone this repo
1. [Get Rye](https://rye-up.com/), this project's package manager.
1. Grab dependencies: `cd training && rye sync`
1. Activate the virtual environment: `. .venv/bin/activate`
    - Other shells may require different commands. For example, in `fish`, you use: `. .venv/bin/activate.fish`.
1. Run the script you'd like to use: `cd src/training && python3 (script).py`

## Instructions

You'll need the following:

- A machine that supports PyTorch
- Rye
- Some time
- The training dataset (ask Barrett over PMs)

Let's begin by downloading the training dataset!

### Dataset

To train the model, you need a dataset. For now, we're using the [Computer Vision Annotation Tool](https://app.cvat.ai) to easily compile our dataset.

1. Download it by going to the SoRo team and hitting the "YOLOv8 for Autonomous" project.
1. In the project menu (three vertical dots), hit "Export dataset".
1. Under the export format box, scroll to the bottom and select "YOLO v1.1".
1. Select "OK".

You should get a download now!

Since you also need a configuration file for the dataset, you'll have to write one manually. Here's how that looks:

```yaml
path: "/home/barrett/Documents/Rover/yolo/rovereye/training/dataset"
train: "obj_Train_data"
val: "obj_Validation_data"

nc: 1
names: ["orange hammer"]
```

Great! Now, save this file as `dataset.yaml`, outside the `dataset/` folder. Change `train.py` to point to `dataset.yaml` (`results = model.train(data="dataset.yaml", epochs=50, imgsz=640)`) so it trains on the correct dataset.

You'll also need to collect the training/validation images. Since we don't have the 'premium' version of CVAT, I can't offer the images alone. You'll need to collect them from those who took them.

Anyway, dump the images in their respective folders. For example, `dataset/obj_Train_data/IMG_1447.txt` also needs a `dataset/obj_Train_data/IMG_1447.JPG`. If a bounding box text file doesn't have an image, remove it from either `dataset/Train.txt` or `dataset/Validation.txt` and remove the `IMG_ABCD.txt` file.

### Training

Now that you have the dataset, consider what options you're going to train with. You can find a full list in Ultralytics' [YOLOv8 documentation](https://docs.ultralytics.com/modes/train/).

Some useful options include:

- `data: str` - the path to a dataset configuration file. That's the `dataset.yaml` that we made before.
- `epochs: u32` - the number of epochs to train for. Defaults to 100.
- `resume: bool` - whether to continue training from where it left off. Defaults to `false`.
  - Note: You'll need to provide the path to the unfinished model if you use this. Ex: `model = YOLO("path/to/last.pt")`
- `save_period: i32` - saves a checkpoint every `save_period` epochs. If it's -1, the default, then this option is disabled.  
- `val: bool` - specifies if validation image training is enabled. Defaults to `true` - and we should probably leave it there. 😄️

After you configure these options in the `train.py` file, navigate to its parent folder and run it using: `python3 train.py`.

### Conversion

After you train the model, you'll need to export it into the ONNX format, which is OpenCV's preferred format.

You only need to run `python3 export.py` to get this model.

### Running

You can now run the Rust project! Go ahead and `cd` back to `rovereye/`. Then, use `cargo run -- -i (input).png` to run an image through the model.

For additional options, you can use `cargo run -- --help` to see the program's help screen. Please let me know if you have any difficulties!
