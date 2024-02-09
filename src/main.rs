use std::env::current_dir;

use anyhow::Context;
use clap::Parser;
use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8, // YOLOv8 by Ultralytics.
};

use opencv::{
    core::{Rect_, Scalar, Vector},
    dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    imgcodecs::imread,
    imgproc::LINE_4,
};

// FIXME(note): our current model ONLY has the orange mallet in its training data lol
const CLASSES_LABELS: [&str; 1] = ["orange mallet"];
const MODEL_NAME: &str = "yolov8_m";

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The input image to scan.
    /// TODO: support video
    #[clap(short, long, default_value = "images/flowers.png")]
    input: String,

    /// The output **location** of the generated file.
    #[clap(short, long, default_value = "images/")]
    output: String,

    /// Logging level for the `tracing` crate.
    /// Values include TRACE, DEBUG, INFO, WARN, and ERROR.
    #[clap(short, long, default_value = "DEBUG")]
    logging_level: tracing::Level,
}

fn main() -> anyhow::Result<()> {
    // parse command line args
    let args = Args::parse();

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
        // create bounding box
        opencv::imgproc::rectangle(
            &mut frame,
            *bbox,
            Scalar::from((0.0, 255.0, 0.0)), //
            2,
            LINE_4,
            0,
        )?;

        // place background behind text + confidence box
        let size = opencv::imgproc::get_text_size(CLASSES_LABELS[class_ids[i]], 2, 1.0, 1, &mut 0)?;
        opencv::imgproc::rectangle(
            &mut frame,
            Rect_::new(
                bbox.x,
                bbox.y - size.height - 5,
                size.width,
                size.height + 5,
            ),
            Scalar::from((0.0, 255.0, 0.0)),
            -1, // fill the background
            LINE_4,
            0,
        )?;

        // place classification text on background
        opencv::imgproc::put_text(
            &mut frame,
            CLASSES_LABELS[class_ids[i]],
            opencv::core::Point::new(bbox.x, bbox.y - 5),
            2,
            1.0,
            Scalar::from((0.0, 0.0, 0.0)),
            1,
            0,
            false,
        )?;

        // confidence text
        let confidence = format!(
            "{:.2}%",
            confidences.get(i).context("guesses have confidence")? * 100_f32
        );
        opencv::imgproc::put_text(
            &mut frame,
            &confidence,
            opencv::core::Point::new(bbox.x + 2, bbox.y + 18),
            1,
            1.0,
            Scalar::from((0.0, 255.0, 0.0)),
            2,
            0,
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
