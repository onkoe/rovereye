use std::fs::create_dir;

use crate::{bb, MODEL_NAME};
use od_opencv::model_ultralytics::ModelUltralyticsV8;
use opencv::{
    imgcodecs::imwrite,
    videoio::{VideoCapture, VideoCaptureTrait as _, VideoCaptureTraitConst as _, CAP_ANY},
};

/// Draws bounding boxes on each frame of the given video.
pub fn process_video(
    input_file: String,
    output_filename: &str,
    output_path: String,
    mut model: ModelUltralyticsV8,
) -> anyhow::Result<()> {
    tracing::debug!("Input {} was detected to be a video.", input_file);

    let mut c = VideoCapture::from_file_def(&input_file)?;

    if !c.is_opened()? {
        tracing::error!("Capture: Couldn't open the given video file.");
        anyhow::bail!("Given video file, {input_file}, may be damaged or malformed.");
    }

    // create output folder
    let output_folder = format!("{0}/{output_filename}", output_path);
    tracing::debug!("Kind::Video: writing video to output folder: `{output_folder}`");
    let _ = create_dir(output_folder);

    let mut v = Vec::new();
    let mut i: u32 = 0;

    // while we have frames, check if they're good and write them to the vec
    while let Ok(true) = c.grab() {
        let mut frame = opencv::core::Mat::default();

        if let Ok(true) = c.retrieve(&mut frame, CAP_ANY) {
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
                output_path
            ),
            frame,
            &opencv::core::Vector::<i32>::new(),
        )?;
    }

    Ok(())
}
