use std::{env::current_dir, path::Path};

use anyhow::Context;
use clap::{Parser, Subcommand};
use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8, // YOLOv8 by Ultralytics.
};

use opencv::{
    core::{Mat, Rect_, Scalar, Vector},
    dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    imgcodecs::imread,
    imgproc::LINE_4,
};

mod bb;

// FIXME(note): our current model ONLY has the orange mallet in its training data lol
const CLASSES_LABELS: [&str; 1] = ["orange mallet"];
const MODEL_NAME: &str = "yolov8_m";

/// Command line arguments for users to pass in.
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

    /// A specific subcommand (task) that should be focused on.
    #[clap(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Takes a continuous input of video and use YOLO on the camera.
    Stream {
        /// Path to a video capture device (like a webcam).
        #[clap(short, long)]
        input: String,
    },
}

fn main() -> anyhow::Result<()> {
    // parse command line args
    let args = Args::parse();

    // enable logging
    let subscriber = tracing_subscriber::fmt()
        .with_file(true)
        .with_line_number(true)
        .compact()
        .with_max_level(args.logging_level)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    // parse input + output files
    let output_filename = Path::new(&args.input)
        .file_stem()
        .context("input files should have names")?
        .to_str()
        .context("input files should have valid UTF-8 characters")?;

    let input_file: String = args.input.to_owned();
    let output_file = format!("{0}/{output_filename}_{MODEL_NAME}.jpg", args.output);

    // FIXME: no clue why this is necessary
    let conf = current_dir()?.to_string_lossy().to_string() + "/config.ini";
    let net_size = (640, 640); // model internal image size

    // create the model
    let mut model = ModelUltralyticsV8::new_from_file(
        "pretrained/last.onnx",
        Some(&conf),
        net_size,
        ModelFormat::ONNX,  // use onnx model
        DNN_BACKEND_OPENCV, // <---- target cpu
        DNN_TARGET_CPU,
        vec![], // no filtered 'classes' of objs
    )?;

    // feed it an example image
    let mut frame = imread(&input_file, 1)?;
    draw_bounding_boxes(&mut model, &mut frame)?;

    // write the bounding boxes
    opencv::imgcodecs::imwrite(&output_file, &frame, &Vector::new())?;
    Ok(())
}

    let input_file: String = args.input.to_owned();
    let input_filetype = file_format::FileFormat::from_file(Path::new(&args.input))?;

    match input_filetype.kind() {
        Kind::Image => {
            tracing::debug!("Input {} was detected to be an image.", input_file);

            // make frame and draw bounding boxes
            let mut frame = imread(&input_file, 1)?;
            bb::draw_bounding_boxes(&mut model, &mut frame)?;

            // export drawn-on frame as output file
            let output_file = format!("{0}/{output_filename}_{MODEL_NAME}.jpg", args.output);
            opencv::imgcodecs::imwrite(&output_file, &frame, &Vector::new())?;
        }

        Kind::Video => {
            tracing::debug!("Input {} was detected to be a video.", input_file);
            // TODO for frame in input_file {}
        }

        k => {
            tracing::error!("Unexpected filetype.");
            anyhow::bail!("Files of type {:?} are not permitted for use in OpenCV.", k);
        }
    }

    Ok(())
}
