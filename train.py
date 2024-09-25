from ultralytics import YOLO
import wandb
import argparse
import time 
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='wandb run name')
    args = parser.parse_args()
    # Initialize a Weights & Biases run

    wandb.init(project="yolo", job_type="training", name=args.name)

    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from scratch
    
    model = YOLO('yolov8n.pt')


    #output = model('/workspace/datasets/val2017/000000000139.jpg')  # ウォームアップ

    #start_time = time.time()
    #with torch.no_grad():
    #    for _ in tqdm(range(5000)):
    #        model('/workspace/datasets/val2017/000000000139.jpg')
    #end_time = time.time()
    #print(f"推論時間（枝刈り前）: {(end_time - start_time) /5000:.4f} 秒")

    # Add W&B Callback for Ultralytics
    train_results = model.train(
        data="coco128.yaml",  # path to dataset YAML
        epochs=150,  # number of training epochs
        imgsz=640,  # training image size
        device="1",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        resume=False,
    )


    # ゼロ比率を表示
    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )
    print('-'*100)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")  # 永続化

    # /workspace/runs/detect/train/weights/best.onnxのような形で保存される
    model.export(format='onnx')  # export the model to ONNX format


    #output = model('/workspace/datasets/val2017/000000000139.jpg')  # ウォームアップ

    #start_time = time.time()
    #with torch.no_grad():
    #    output = model('/workspace/datasets/val2017/000000000139.jpg')
    #end_time = time.time()
    #print(f"推論時間（永続化後）: {end_time - start_time:.4f} 秒")


    # Intel Core i7-12700 で 0.0270 秒
    # マスクをパラメータに永続化したのでオーバーヘッドはなくなるが、疎計算に対応していないので、密計算と同じ速度

    #wandb.finish()

    