use thiserror::Error;

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

