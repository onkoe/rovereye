use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::bb;
use od_opencv::model_ultralytics::ModelUltralyticsV8;
use opencv::core::Mat;
use opencv::highgui::{imshow, named_window, WINDOW_NORMAL};
use opencv::hub_prelude::MatTraitConst;
use opencv::prelude::VideoCaptureTraitConst;
use opencv::videoio::{
    self, VideoCapture, VideoCaptureTrait, CAP_ANY, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
};
use tracing::instrument;

/// Draws bounding boxes on each frame received from the capture device (stream).
#[instrument(skip(model))]
pub fn stream(camera: &mut VideoCapture, model: &mut ModelUltralyticsV8) -> anyhow::Result<()> {
    let mut frame = Mat::default();
    let mut timer = Instant::now();

    tracing::debug!("attempting to retrieve images...");
    sleep(Duration::from_millis(500));

    while let Ok(true) = camera.grab() {
        if let Ok(true) = camera.retrieve(&mut frame, videoio::CAP_ANY) {
            if frame.size()?.empty() {
                tracing::debug!("no image was captured!");
                continue;
            }

            named_window("SoRo Autonomous - YOLOv8 Stream Mode", WINDOW_NORMAL)?;

            // process frame
            bb::draw_bounding_boxes(model, &mut frame)?;

            // display frame
            imshow("SoRo Autonomous - YOLOv8 Stream Mode", &frame)?;
            opencv::highgui::wait_key(1)?;

            // let user know that things are still happening!
            if timer.elapsed() > Duration::from_secs(4) {
                tracing::debug!(
                    "still using the camera. timer offset: {:?}",
                    timer.elapsed()
                );

                timer = Instant::now();
            }
        } else {
            tracing::error!("couldn't get new camera image!");
        }
    }

    Ok(())
}

/// Given a capture device, sets various props and returns the modified device.
pub fn set_capture_device(
    mut cam: VideoCapture,
    cam_path: String,
) -> Result<VideoCapture, anyhow::Error> {
    cam.set(CAP_PROP_FRAME_WIDTH, super::NET_SIZE.0 as f64)?;
    cam.set(CAP_PROP_FRAME_HEIGHT, super::NET_SIZE.1 as f64)?;

    sleep(Duration::from_millis(500)); // wait for the cam to stabilize
    tracing::info!("Capture device found!");

    if !cam.is_opened()? {
        tracing::error!("Unable to open given capture device!");
        anyhow::bail!("Failed to open capture device at: `{cam_path:?}`",);
    }

    Ok(cam)
}

/// Creates a new capture device from command line args.
pub fn create_capture_device_from_args(
    input: Option<String>,
    device: Option<i32>,
) -> anyhow::Result<VideoCapture> {
    // avoid passing both or none!
    if input.is_some() && device.is_some() || input.is_none() && device.is_none() {
        tracing::error!("Please pass either an `input` or a `device` argument.");
        anyhow::bail!("Malformed arguments.");
    }

    if let Some(path_str) = input {
        let path = Path::new(&path_str);

        // see if path exists
        if path.exists() {
            tracing::info!("Attempting to use capture device!");

            return Ok(opencv::videoio::VideoCapture::from_file_def(&path_str)?);
        }
    }

    // check if device id works
    if let Some(device) = device {
        tracing::info!("Attempting to use capture device!");

        let cam: VideoCapture = opencv::videoio::VideoCapture::new(device, CAP_ANY)?;
        return set_capture_device(cam, device.to_string());
    }

    anyhow::bail!("No capture device found.");
}
