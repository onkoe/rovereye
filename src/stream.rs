use std::thread::sleep;
use std::time::{Duration, Instant};

use opencv::core::Mat;
use opencv::highgui::{imshow, named_window, WINDOW_NORMAL};
use opencv::hub_prelude::MatTraitConst;
use opencv::videoio::{self, VideoCapture, VideoCaptureTrait};
use tracing::instrument;

#[instrument]
pub fn stream(camera: &mut VideoCapture) -> anyhow::Result<()> {
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

            named_window("SoRO Autonomous - YOLOv8 Stream Mode", WINDOW_NORMAL)?;
            imshow("SoRO Autonomous - YOLOv8 Stream Mode", &frame)?; // TODO: yolo!
            opencv::highgui::wait_key(1)?;

            if timer.elapsed() > Duration::from_secs(4) {
                tracing::debug!(
                    "still using the camera. timer offset: {:?}",
                    timer.elapsed()
                );

                timer = Instant::now();
            }

            sleep(Duration::from_millis(16));
        } else {
            tracing::error!("couldn't get new camera image!");
        }
    }

    Ok(())
}
