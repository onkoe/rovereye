use std::{env::current_dir, path::Path};

use anyhow::Context;
use clap::{Parser, Subcommand};
use file_format::Kind;
use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8, // YOLOv8 by Ultralytics.
};
use opencv::dnn::{DNN_BACKEND_CUDA, DNN_BACKEND_OPENCV, DNN_TARGET_CPU, DNN_TARGET_CUDA};

mod bounding_box;
mod image;
mod stream;
mod video;

pub const CLASSES_LABELS: [&str; 1] = ["orange mallet"];
pub const MODEL_NAME: &str = "yolov8_m";
pub const NET_SIZE: (i32, i32) = (640, 640); // model internal image size

/// Command line arguments for users to pass in.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The input image to scan.
    #[clap(short, long, default_value = "images/flowers.png")]
    input: String,

    /// The output location of the generated file.
    #[clap(short, long, default_value = "images/")]
    output: String,

    /// Logging level for the `tracing` crate.
    /// Values include TRACE, DEBUG, INFO, WARN, and ERROR.
    #[clap(short, long, default_value = "DEBUG")]
    logging_level: tracing::Level,

    /// The pretrained model to use during detection.
    #[clap(short, long, default_value = "pretrained/best.onnx")]
    model: String,

    /// A specific subcommand (task) that should be focused on.
    #[clap(subcommand)]
    command: Option<Command>,

    /// Toggles CUDA support.
    #[clap(short, long, action)]
    cuda: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Takes a continuous input of video and use YOLO on the camera.
    Stream {
        /// Path to a video capture device (like a webcam).
        #[clap(short, long)]
        input: Option<String>,
        /// Video capture device number (used for Windows and macOS).
        /// If no device is specified, uses `input` instead.
        #[clap(short, long)]
        device: Option<i32>,
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

    // create the model
    let mut model = create_model(&args.model, args.cuda)?;

    // handle streaming subcommand... if necessary
    if let Some(Command::Stream { input, device }) = args.command {
        let mut cam = stream::create_capture_device_from_args(input, device)?;
        stream::stream(&mut cam, &mut model)?;
    }

    // parse input + output files
    let output_filename = Path::new(&args.input)
        .file_stem()
        .context("input files should have names")?
        .to_str()
        .context("input files should have valid UTF-8 characters")?;

    let input_file: String = args.input.to_owned();
    let input_filetype = file_format::FileFormat::from_file(Path::new(&args.input))?;
    tracing::debug!("input_file: {input_file}");

    match input_filetype.kind() {
        Kind::Image => image::process_image(input_file, output_filename, args.output, model)?,
        Kind::Video => video::process_video(input_file, output_filename, args.output, model)?,
        x => {
            tracing::error!("Unexpected filetype.");
            anyhow::bail!("Files of type {:?} are not permitted for use in OpenCV.", x);
        }
    }

    Ok(())
}

/// Given a path to a pre-trained model, attempts to initalize a model object.
pub fn create_model(model_path: &str, cuda_support: bool) -> anyhow::Result<ModelUltralyticsV8> {
    // FIXME: no clue why this is necessary
    let conf = current_dir()?.to_string_lossy().to_string() + "/config.ini";

    // check for CUDA flag. CPU is the default
    let (backend, target) = if cuda_support {
        tracing::info!("CUDA usage requested. Using CUDA backend.");
        (DNN_BACKEND_CUDA, DNN_TARGET_CUDA)
    } else {
        (DNN_BACKEND_OPENCV, DNN_TARGET_CPU)
    };

    Ok(ModelUltralyticsV8::new_from_file(
        model_path,
        Some(&conf),
        NET_SIZE,
        ModelFormat::ONNX, // use onnx model
        backend,
        target,
        vec![],
    )?)
}
