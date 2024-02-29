use anyhow::Context;
use od_opencv::model_ultralytics::ModelUltralyticsV8;
use opencv::{
    core::{Mat, Rect_, Scalar},
    imgproc::LINE_4,
};

use crate::CLASSES_LABELS;

/// Given the the model and frame, utilizes the model's insights to create
/// bounding boxes around located objects.
pub fn draw_bounding_boxes(model: &mut ModelUltralyticsV8, frame: &mut Mat) -> anyhow::Result<()> {
    // use the model on the image
    let (bboxes, class_ids, confidences) = model.forward(frame, 0.25, 0.4)?;

    // draw the boxes!
    for (i, bbox) in bboxes.iter().enumerate() {
        // create the bounding box
        opencv::imgproc::rectangle(
            frame,
            *bbox,
            Scalar::from((0.0, 255.0, 0.0)), //
            2,
            LINE_4,
            0,
        )?;

        // place background behind text + confidence box
        let size = opencv::imgproc::get_text_size(CLASSES_LABELS[class_ids[i]], 2, 1.0, 1, &mut 0)?;
        opencv::imgproc::rectangle(
            frame,
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
            frame,
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
            frame,
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
        tracing::trace!("Bounding box #{i} position: {:?}", bbox);
    }
    Ok(())
}
