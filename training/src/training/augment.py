from typing import List

import albumentations
import cv2
import datumaro
from loguru import logger

# load all the bounding boxes
data = datumaro.Dataset.import_from(path="data/export_dataset_feb_28_2024/data.yaml")

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
    bbox_params=albumentations.BboxParams(format="coco"),
)

for image in data:
    logger.info(f"Processing image {image.attributes.get('path')}")
    
    # load from disk
    loaded_image = cv2.imread(image.attributes.get("path"))
    loaded_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply the augmentations
    augmented = transform(image=image, bboxes=image.annotations)
    augmented_image = augmented["image"]
    augmented_bboxes = augmented["bboxes"]

    # save the augmented image
    cv2.imwrite("augmented_aug.jpg", augmented)
    
    # push the bounding boxes to the dataset
    data.put(augmented_bboxes)
    
# write augmented dataset to disk
data.export("augmented_dataset.yaml", format="coco")