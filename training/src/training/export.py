from ultralytics import YOLO

model = YOLO("../../../pretrained/yolov8n.pt")

# export to ONNX format for use with OpenCV
model.export(format="onnx", opset=12)
