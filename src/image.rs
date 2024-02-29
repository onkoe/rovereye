use crate::{bb, MODEL_NAME};
use od_opencv::model_ultralytics::ModelUltralyticsV8;
use opencv::{core::Vector, imgcodecs::imread};

/// Draws bounding boxes on the given image.
pub fn process_image(
    input_file: String,
    output_filename: &str,
    output_path: String,
    mut model: ModelUltralyticsV8,
) -> anyhow::Result<()> {
    tracing::debug!("Input {} was detected to be an image.", input_file);

    // make frame and draw bounding boxes
    let mut frame = imread(&input_file, 1)?;
    bb::draw_bounding_boxes(&mut model, &mut frame)?;

    // export drawn-on frame as output file
    let output_file = format!("{0}/{output_filename}_{MODEL_NAME}.jpg", output_path);
    opencv::imgcodecs::imwrite(&output_file, &frame, &Vector::new())?;

    Ok(())
}
