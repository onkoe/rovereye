use std::thread::sleep;
use std::time::{Duration, Instant};

use od_opencv::model_ultralytics::ModelUltralyticsV8;
use opencv::core::Mat;
use opencv::highgui::{imshow, named_window, WINDOW_NORMAL};
use opencv::hub_prelude::MatTraitConst;
use opencv::videoio::{self, VideoCapture, VideoCaptureTrait};
use tracing::instrument;

use crate::bb;

#[instrument(skip(model))]
pub fn stream(camera: &mut VideoCapture, model: &mut ModelUltralyticsV8) -> anyhow::Result<()> {
    let mut frame = Mat::default();
    let mut timer = Instant::now();

    tracing::debug!("attempting to retrieve images...");
    sleep(Duration::from_millis(500)); // rust is too fast :o

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
