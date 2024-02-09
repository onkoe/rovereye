from ultralytics import YOLO

best = YOLO("../../../pretrained/best.pt")  # the 'best' model according to val
last = YOLO("../../../pretrained/last.pt")  # the final iteration in training


# export to ONNX format for use with OpenCV
best.export(format="onnx", opset=12)
last.export(format="onnx", opset=12)
