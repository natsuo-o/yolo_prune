from ultralytics import YOLO
from deepsparse import Pipeline

#model = YOLO('/workspace/yolov8n.pt')
#model.export(format='onnx')

# Set up the DeepSparse Pipeline
# yolo_pipeline = Pipeline.create(task="yolo", model_path='/workspace/runs/detect/train/weights/best.onnx')
yolo_pipeline = Pipeline.create(task="yolo", model_path='/workspace/yolov8n.onnx')
# Run the model on your images
images = ["/workspace/datasets/val2017/000000000139.jpg"]
pipeline_outputs = yolo_pipeline(images=images)
start_time = time.time()
for _ in tqdm(range(5000)):
    pipeline_outputs = yolo_pipeline(images=images)
end_time = time.time()
print(f"推論時間（永続化後）: {(end_time - start_time) / 5000:.4f} 秒")

print(pipeline_outputs)

# Intel Core i7-12700 で 0.146 秒
    
