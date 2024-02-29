use std::{env::current_dir, fs::create_dir, path::Path};

use anyhow::Context;
use clap::{Parser, Subcommand};
use file_format::Kind;
use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8, // YOLOv8 by Ultralytics.
};
use opencv::{
    core::Vector,
    dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    imgcodecs::{imread, imwrite},
    videoio::{self, VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst},
};

mod bb;
mod stream;

// FIXME(note): our current model ONLY has the orange mallet in its training data lol
const CLASSES_LABELS: [&str; 1] = ["orange mallet"];
const MODEL_NAME: &str = "yolov8_m";
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
    #[clap(short, long, default_value = "pretrained/best.pt")]
    model: String,

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

    // FIXME: no clue why this is necessary
    let conf = current_dir()?.to_string_lossy().to_string() + "/config.ini";

    // create the model
    let mut model = ModelUltralyticsV8::new_from_file(
        &args.model,
        Some(&conf),
        NET_SIZE,
        ModelFormat::ONNX, // use onnx model
        DNN_BACKEND_OPENCV,
        DNN_TARGET_CPU, // <---- target cpu
        vec![],
    )?;

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

            let mut c = VideoCapture::from_file_def(&input_file)?;

            if !c.is_opened()? {
                tracing::error!("Capture: Couldn't open the given video file.");
                anyhow::bail!("Given video file, {input_file}, may be damaged or malformed.");
            }

            // create output folder
            let output_folder = format!("{0}/{output_filename}", args.output);
            tracing::debug!("Kind::Video: writing video to output folder: `{output_folder}`");
            let _ = create_dir(output_folder);

            let mut v = Vec::new();
            let mut i: u32 = 0;

            // while we have frames, check if they're good and write them to the vec
            while let Ok(true) = c.grab() {
                let mut frame = opencv::core::Mat::default();

                if let Ok(true) = c.retrieve(&mut frame, videoio::CAP_ANY) {
                    tracing::debug!("frame {i}");
                    bb::draw_bounding_boxes(&mut model, &mut frame)?;
                    v.push(frame.clone());
                    i += 1;
                }
            }

            // let's do all the writing now!
            for (i, frame) in v.iter().enumerate() {
                imwrite(
                    &format!(
                        "{0}/{output_filename}/{output_filename}_{i}_{MODEL_NAME}.jpg",
                        args.output
                    ),
                    frame,
                    &opencv::core::Vector::<i32>::new(),
                )?;
            }
        }

        k => {
            tracing::error!("Unexpected filetype.");
            anyhow::bail!("Files of type {:?} are not permitted for use in OpenCV.", k);
        }
    }

    Ok(())
}
