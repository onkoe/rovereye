//! # rovereye
//!
//! A way to find the items requested by the 2024 URC, including the orange
//! mallet and the water bottles.

use std::{env::current_dir, path::Path};

use od_opencv::{model_format::ModelFormat, model_ultralytics::ModelUltralyticsV8};
use opencv::{
    dnn::{DNN_BACKEND_CUDA, DNN_TARGET_CUDA},
    videoio::VideoCapture,
};
use thiserror::Error;

pub const CLASSES_LABELS: [&str; 2] = ["orange mallet", "water bottle"];
pub const MODEL_NAME: &str = "yolov8_m";
pub const NET_SIZE: (i32, i32) = (640, 640); // model internal image size

pub struct Model {
    model: ModelUltralyticsV8,
    camera: VideoCapture,
}

impl Model {
    /// Creates a new model given a camera path (like /dev/video0) and a model
    /// locationcarg (like ~/Downloads/my_model.onnx).
    pub fn new_from_camera_path(camera_path: &str, model_path: &str) -> Result<Model, ModelError> {
        let model = Self::create_yolo_instance(model_path)?;
        let camera = {
            let path = Path::new(camera_path);
            let err = ModelError::NoCaptureDevicePath(camera_path.to_owned());

            if path.exists() {
                VideoCapture::from_file_def(camera_path).map_err(|_| err)?
            } else {
                return Err(err);
            }
        };

        Ok(Self { model, camera })
    }

    /// Given an OpenCV camera ID and model location, creates a new model
    /// instance.
    pub fn new_from_camera_number(
        camera_number: i32,
        model_path: &str,
    ) -> Result<Model, ModelError> {
        let model = Self::create_yolo_instance(model_path)?;
        let camera = VideoCapture::new_def(camera_number)
            .map_err(|_| ModelError::NoCaptureDeviceNumber(camera_number))?;

        Ok(Self { model, camera })
    }

    /// Captures a new image from the camera, then looks for
    pub fn scan(&self) -> Vec<(ItemType, CornerList)> {
        todo!()
    }

    /// Creates a new instance of the YOLO model for internal use.
    fn create_yolo_instance(model_path: &str) -> Result<ModelUltralyticsV8, ModelError> {
        // TODO: no idea why this is necessary
        let conf = current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
            + "/config.ini";

        let (backend, target) = (DNN_BACKEND_CUDA, DNN_TARGET_CUDA);

        ModelUltralyticsV8::new_from_file(
            model_path,
            Some(&conf),
            NET_SIZE,
            ModelFormat::ONNX, // use onnx model
            backend,
            target,
            vec![],
        )
        .map_err(|_| ModelError::ModelPathDoesntExist(model_path.to_owned()))
    }
}

/// An error that can occur when creating a new model.
#[derive(Clone, Debug, PartialEq, PartialOrd, Hash, Error)]
pub enum ModelError {
    #[error("The given capture device (camera) path doesn't exist: `{0}`")]
    NoCaptureDevicePath(String),
    #[error("The specified OpenCV capture device number has no assigned camera: `{0}`")]
    NoCaptureDeviceNumber(i32),
    #[error("The given model path doesn't exist: `{0}`")]
    ModelPathDoesntExist(String),
}

/// The type of item found by the model.
#[derive(Clone, Debug, PartialEq, PartialOrd, Hash)]
#[pyo3::pyclass]
pub enum ItemType {
    OrangeMallet,
    WaterBottle,
}

/// The location of an object on an image, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Hash)]
pub struct Coordinate {
    x: u64,
    y: u64,
}

impl Coordinate {
    /// Returns the `x` value of the coordinate.
    pub fn x(&self) -> u64 {
        self.x
    }

    /// Returns the `y` value of the coordinate.
    pub fn y(&self) -> u64 {
        self.y
    }
}

/// A list of corners found on any object within an image.
/// This creates a bounding box.
#[derive(Clone, Debug, PartialEq, PartialOrd, Hash)]
pub struct CornerList {
    top_left: Coordinate,
    bottom_left: Coordinate,
    top_right: Coordinate,
    bottom_right: Coordinate,
}

impl CornerList {
    /// Returns the top-left corner of the bounding box.
    fn top_left(&self) -> Coordinate {
        self.top_left
    }

    /// Returns the bottom-left corner of the bounding box.
    fn bottom_left(&self) -> Coordinate {
        self.bottom_left
    }

    /// Returns the top-right corner of the bounding box.
    fn top_right(&self) -> Coordinate {
        self.top_right
    }

    /// Returns the bottom-right corner of the bounding box.
    fn bottom_right(&self) -> Coordinate {
        self.bottom_right
    }
}
