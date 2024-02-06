use std::env::current_dir;

use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8, // YOLOv8 by Ultralytics.
};
use opencv::{
    core::{Scalar, Vector},
    dnn::DNN_BACKEND_OPENCV,
    dnn::DNN_TARGET_CPU,
    imgcodecs::imread,
    imgproc::LINE_4,
};

#[rustfmt::skip] // keep this monstrosity outta my eyes
const CLASSES_LABELS: [&str; 80] = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

fn main() -> anyhow::Result<()> {
    // enable logging
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber)?;

    // get config file
    let conf = current_dir()?.to_string_lossy().to_string() + "/config.toml";

    // set expected img size (width x height)
    let net_size = (640, 640);

    // create the model
    let mut model = ModelUltralyticsV8::new_from_file(
        "yolov8n.onnx",
        Some(&conf),
        net_size,
        ModelFormat::ONNX,  // use onnx model
        DNN_BACKEND_OPENCV, // <---- target cpu
        DNN_TARGET_CPU,
        vec![], // no filtered 'classes' of objs
    )?;

    // feed it an example image
    let mut frame = imread("images/dog.jpg", 1)?;

    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
    for (i, bbox) in bboxes.iter().enumerate() {
        opencv::imgproc::rectangle(
            &mut frame,
            *bbox,
            Scalar::from((0.0, 255.0, 0.0)), //
            2,
            LINE_4,
            0,
        )?;
        tracing::trace!("bounding box #{i}...");

        tracing::debug!("Class: {}", CLASSES_LABELS[class_ids[i]]);
        tracing::debug!("\tBounding box: {:?}", bbox);
        tracing::debug!("\tConfidences: {}", confidences[i]);
    }

    // write the bounding boxes
    opencv::imgcodecs::imwrite("images/dog_yolov8_n.jpg", &frame, &Vector::new())?;

    Ok(())
}