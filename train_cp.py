from ultralytics import YOLO
# import wandb
# from wandb.integration.ultralytics import add_wandb_callback
import argparse
import time 
import torch
import torch.nn.utils.prune as prune
from ultralytics.utils.benchmarks import benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='wandb run name')
    args = parser.parse_args()

    # Initialize a Weights & Biases run
    # wandb.init(project="yolo_pruned", job_type="training", name=args.name)

    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model = YOLO('yolov8n.pt')
    model2 = torch.load('yolov8n.pt')
    model3 =torch.load('yolov8n.pt')['model']
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    for module in model.model.modules():
        print(1)
        if isinstance(module, torch.nn.Conv2d):
            print(module)
            print(module.weight)
            print((module.weight[0], "weight"))
            break
    
    for module in model2['model'].modules():
        print(2)
        if isinstance(module, torch.nn.Conv2d):
            print((module.weight[0], "weight"))
            break
    for module in model3.modules():
        print(3)
        if isinstance(module, torch.nn.Conv2d):
            print((module.weight[0], "weight"))
            break

    # Add W&B Callback for Ultralytics
    # add_wandb_callback(model, enable_model_checkpointing=True)

    # ウォームアップ
    # model.eval()
    # output = model.predict('/workspace/datasets/val2017/000000000139.jpg')

    #start_time = time.time()

    #with torch.no_grad():
    #    output = model('/workspace/datasets/val2017/000000000139.jpg')
    #end_time = time.time()
    #print(f"推論時間（密）: {end_time - start_time:.4f} 秒")
    parameters_to_prune = [
    (module, "weight") for module in model.model.modules() if isinstance(module, torch.nn.Conv2d)
    ]  # すべての畳み込み層を枝刈り対象にする


    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.8,
    )  # 大域的・非構造・強度枝刈り
    # metrics = model.val(data="coco8.yaml", device='cpu') 
    parameters_to_prune2 = [
    (module, "weight") for module in model2['model'].modules() if isinstance(module, torch.nn.Conv2d)
    ]  # すべての畳み込み層を枝刈り対象にする


    prune.global_unstructured(
        parameters_to_prune2,
        pruning_method=prune.L1Unstructured,
        amount=0.1,
    )  # 大域的・非構造・強度枝刈り


    parameters_to_prune3 = [
        (module, "weight") for module in model3.modules() if isinstance(module, torch.nn.Conv2d)
        ]  # すべての畳み込み層を枝刈り対象にする


    prune.global_unstructured(
        parameters_to_prune3,
        pruning_method=prune.L1Unstructured,
        amount=0.1,
    )  # 大域的・非構造・強度枝刈り

    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )
    print('-'*100)

    for module in model2['model'].modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )
    print('-'*100)

    for module in model3.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )
    print('-'*100)

    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(4)
            print(module)
            print((module.weight[0], "weight"))
            break
    print('9'*100)
    for module in model2['model'].modules():
        if isinstance(module, torch.nn.Conv2d):
            print(module)
            print(5)
            print((module.weight[0], "weight"))
            break
    for module in model3.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(module)
            print(6)
            print((module.weight[0], "weight"))
            break

    # ウォームアップ
    #model('/workspace/datasets/val2017/000000000139.jpg')  # ウォームアップ

    #start_time = time.time()
    #with torch.no_grad():
    #    output = model('/workspace/datasets/val2017/000000000139.jpg')
    #end_time = time.time()
    #print(f"推論時間（枝刈り直後）: {end_time - start_time:.4f} 秒")
    # Intel Core i7-12700 で 0.0359 秒
    # マスクを都度適用しているのでかえって遅くなる

    print(len(model.model.float().state_dict()))

    # Use the model
    model4 = YOLO('yolov8n.pt')

    #print(model.model.float().state_dict())
    #print(len(model.model.float().state_dict()))


    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )
    print('-'*100)



    train_results = model4.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=10,  # number of training epochs
        imgsz=640,  # training image size
        device="1",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        resume=False
    )

    #for module in model.modules():
    #    if isinstance(module, torch.nn.Conv2d):
    #        print((module.weight[0], "weight"))
    #        c
    #model = YOLO('/workspace/runs/detect/train19/weights/last.pt')  # load a pretrained model (recommended for training)


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


    output = model('/workspace/datasets/val2017/000000000139.jpg')  # ウォームアップ

    start_time = time.time()
    with torch.no_grad():
        output = model('/workspace/datasets/val2017/000000000139.jpg')
    end_time = time.time()
    print(f"推論時間（永続化後）: {end_time - start_time:.4f} 秒")

    c
    # Intel Core i7-12700 で 0.0270 秒
    # マスクをパラメータに永続化したのでオーバーヘッドはなくなるが、疎計算に対応していないので、密計算と同じ速度


    # Validate the model
    metrics = model.val(data="coco8.yaml", )  # no arguments needed, dataset and settings remembered
    #metrics.box.map  # map50-95
    #metrics.box.map50  # map50
    #metrics.box.map75  # map75
    #metrics.box.maps  # a list contains map50-95 of each category

    #wandb.finish()

    results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    print()


    results = model.export(format='onnx')  # export the model to ONNX format