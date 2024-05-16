from typing import List
import albumentations
import datumaro
import cv2

# load all the bounding boxes
boxes: List[List[float]] = []

# create the list of augmentations to perform
transform = albumentations.Compose(
    [
        albumentations.RandomCrop(width=640, height=640, p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.OneOf(
            [
                albumentations.Blur(blur_limit=3, p=0.5),
                albumentations.MedianBlur(blur_limit=3, p=0.5),
                albumentations.GaussNoise(p=0.5),
                albumentations.MotionBlur(blur_limit=3, p=0.5),
                albumentations.OpticalDistortion(p=0.5),
                albumentations.GridDistortion(p=0.5),
                albumentations.ElasticTransform(p=0.5),
                albumentations.CLAHE(p=0.5),
            ],
            p=0.5,
        ),
    ],
    bbox_params=albumentations.BboxParams(format="yolo"),
)
