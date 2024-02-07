from ultralytics import YOLO

# import pretrained model
model = YOLO("../../../pretrained/yolov8n.pt")

# train the model
# to insert our own data, replace "coco128.yaml" with "().yaml"
results = model.train(data="coco128.yaml", epochs=50, imgsz=640)
