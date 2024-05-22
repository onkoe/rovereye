//! # rovereye
//!
//! A way to find the items requested by the 2024 URC, including the orange
//! mallet and the water bottles.

use std::{env::current_dir, path::Path};

use od_opencv::{model_format::ModelFormat, model_ultralytics::ModelUltralyticsV8};
use opencv::{
    core::{Mat, MatTraitConst as _, Rect_},
    dnn::{DNN_BACKEND_CUDA, DNN_TARGET_CUDA},
    videoio::{VideoCapture, VideoCaptureTrait as _, CAP_ANY},
};

pub const CLASSES_LABELS: [&str; 2] = ["orange mallet", "water bottle"];
pub const MODEL_NAME: &str = "yolov8_m";
pub const NET_SIZE: (i32, i32) = (640, 640); // model internal image size

use error::ModelError;
use pyo3::{PyErr, PyResult};

mod error;

#[pyo3::pyclass]
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

    /// Captures a new image from the camera, then looks for the items. Use
    /// this sparingly - each call grabs a new image and creates bounding boxes.
    pub fn scan(&mut self) -> Option<(Vec<CornerList>, Vec<ItemType>, Vec<f32>)> {
        let mut frame = Mat::default();

        if let Ok(true) = self.camera.grab() {
            if let Ok(true) = self.camera.retrieve(&mut frame, CAP_ANY) {
                if frame.size().ok()?.empty() {
                    tracing::debug!("no image was captured!");
                    return None;
                }

                let (bboxes, class_ids, confidences) =
                    self.model.forward(&frame, 0.25, 0.4).ok()?;

                let types = class_ids
                    .iter()
                    .filter_map(|&v| match v {
                        0 => Some(ItemType::OrangeMallet),
                        1 => Some(ItemType::WaterBottle),
                        _ => {
                            tracing::error!("Detected an unknown item.");
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                let bounding_boxes = bboxes
                    .iter()
                    .map(|&b| b.into())
                    .collect::<Vec<CornerList>>();

                return Some((bounding_boxes, types, confidences));
            }
        }

        None
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

#[pyo3::pymethods]
impl Model {
    #[pyo3(name = "new_from_camera_path")]
    pub fn py_new_from_camera_path(&self, camera_path: &str, model_path: &str) -> PyResult<Self> {
        Self::new_from_camera_path(camera_path, model_path).map_err(PyErr::from)
    }

    #[pyo3(name = "new_from_camera_number")]
    pub fn py_new_from_camera_number(
        &self,
        camera_number: i32,
        model_path: &str,
    ) -> PyResult<Self> {
        Self::new_from_camera_number(camera_number, model_path).map_err(PyErr::from)
    }

    #[pyo3(name = "scan")]
    pub fn py_scan(&mut self) -> Option<(Vec<CornerList>, Vec<ItemType>, Vec<f32>)> {
        self.scan()
    }
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
#[pyo3::pyclass]
pub struct Coordinate {
    x: i32,
    y: i32,
}

impl Coordinate {
    /// Creates a new coordinate.
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Returns the `x` value of the coordinate.
    pub fn x(&self) -> i32 {
        self.x
    }

    /// Returns the `y` value of the coordinate.
    pub fn y(&self) -> i32 {
        self.y
    }
}

/// A list of corners found on any object within an image.
/// This creates a bounding box.
#[derive(Clone, Debug, PartialEq, PartialOrd, Hash)]
#[pyo3::pyclass]
pub struct CornerList {
    top_left: Coordinate,
    bottom_left: Coordinate,
    top_right: Coordinate,
    bottom_right: Coordinate,
}

#[pyo3::pymethods]
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

impl From<Rect_<i32>> for CornerList {
    fn from(value: Rect_<i32>) -> Self {
        Self {
            top_left: Coordinate::new(value.x, value.y),
            top_right: Coordinate::new(value.x + value.width, value.y),
            bottom_left: Coordinate::new(value.x, value.y + value.height),
            bottom_right: Coordinate::new(value.x + value.width, value.y + value.height),
        }
    }
}
