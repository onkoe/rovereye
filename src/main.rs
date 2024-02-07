use std::env::current_dir;

use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8, // YOLOv8 by Ultralytics.
};

use opencv::{
    core::{Scalar, Vector},
    dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    imgcodecs::imread,
    imgproc::{LINE_4, LINE_8},
};
use tracing::Level;

#[rustfmt::skip] // keep this monstrosity outta my eyes
const CLASSES_LABELS: [&str; 80] = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

fn main() -> anyhow::Result<()> {
    // enable logging
    let subscriber = tracing_subscriber::fmt()
        .with_file(true)
        .with_line_number(true)
        .compact()
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // get config file (./config.ini; i dont know what this does lmao)
    let conf = current_dir()?.to_string_lossy().to_string() + "/config.ini";

    // set expected img size (width x height)
    let net_size = (640, 640);

    // create the model
    let mut model = ModelUltralyticsV8::new_from_file(
        "pretrained/yolov8n.onnx",
        Some(&conf),
        net_size,
        ModelFormat::ONNX,  // use onnx model
        DNN_BACKEND_OPENCV, // <---- target cpu
        DNN_TARGET_CPU,
        vec![], // no filtered 'classes' of objs
    )?;

    // feed it an example image
    let mut frame = imread("images/flowers.png", 1)?;

    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4)?;

    for (i, bbox) in bboxes.iter().enumerate() {
        opencv::imgproc::rectangle(
            &mut frame,
            *bbox,
            Scalar::from((0.0, 255.0, 0.0)), //
            2,
            LINE_4,
            0,
        )?;

        opencv::imgproc::put_text(
            &mut frame,
            CLASSES_LABELS[class_ids[i]],
            opencv::core::Point::new(bbox.x, bbox.y - 5),
            1,
            1.8,
            Scalar::from((0.0, 255.0, 0.0)),
            2,
            LINE_8,
            false,
        )?;

        tracing::debug!(
            "[Classification #{i}] Found `{}` (confidence: {confidence}).",
            CLASSES_LABELS[class_ids[i]]
        );
        tracing::trace!("Bounding box: {:?}", bbox);
    }

    // write the bounding boxes
    opencv::imgcodecs::imwrite("images/flowers_yolov8_n.jpg", &frame, &Vector::new())?;

    Ok(())
}
