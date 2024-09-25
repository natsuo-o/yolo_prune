import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import torch.nn.utils.prune as prune
import torchvision.models as models
from torch.onnx import export
from torchvision import datasets, transforms


def model_to_onnx(
    model,
    output_file,
    input_shape=(1, 3, 224, 224),
):
    model.eval()
    input_tensor = torch.randn(input_shape)
    input_names = ["input"]
    output_names = ["output"]
    export(model, input_tensor, output_file, verbose=False, input_names=input_names, output_names=output_names)
    return output_file


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def change_layers(model):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, 10, bias=True)
    return model


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = models.resnet50(weights="IMAGENET1K_V2")


    model = change_layers(model).to('cuda')
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)



    parameters_to_prune = [
        (module, "weight") for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]  # すべての畳み込み層を枝刈り対象にする

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.99,
    )  # 大域的・非構造・強度枝刈り

    #for module in model.modules():
    #    if isinstance(module, torch.nn.Conv2d):
    #        prune.remove(module, "weight")  # 永続化(重み*maskの計算をしたものをmaskとして扱う)


    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    model2 = models.resnet50(weights="IMAGENET1K_V2")


    model2 = change_layers(model2).to('cuda')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train(args, model2, device, train_loader, optimizer, epoch)

        
        #test(model, device, test_loader)
        scheduler.step()

    #for module in model.modules():
    #    if isinstance(module, torch.nn.Conv2d):
    #        prune.remove(module, "weight")  # 永続化


    # ゼロ比率を表示
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            print(
                f"{module} Zero-Ratio: {100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%"
            )

    parameters_to_prune2 = [
        (module, "weight") for module in model2.modules() if isinstance(module, torch.nn.Conv2d)
    ]  # すべての畳み込み層を枝刈り対象にする

    prune.global_unstructured(
        parameters_to_prune2,
        pruning_method=prune.L1Unstructured,
        amount=0.99,
    )  # 大域的・非構造・強度枝刈り

    for module, module2 in zip(model.modules(), model2.modules()):
        if isinstance(module, torch.nn.Conv2d):
            print(module)
            print((module.weight[0], "weight"))
        if isinstance(module2, torch.nn.Conv2d):
            print(module2)
            print((module2.weight[0], "weight"))
            break





    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

'''
    # モデルを読み込む
model = models.resnet50(weights="IMAGENET1K_V2")
model_to_onnx(model, "resnet50_dense.onnx") # 密モデルを ONNX 形式で保存

input_image = torch.ones((1, 3, 224, 224))
output = model(input_image)  # ウォームアップ

start_time = time.time()
with torch.no_grad():
    output = model(input_image)
end_time = time.time()
print(f"推論時間（密）: {end_time - start_time:.4f} 秒")
# Intel Core i7-12700 で 0.0277 秒

parameters_to_prune = [
    (module, "weight") for module in model.modules() if isinstance(module, torch.nn.Conv2d)
]  # すべての畳み込み層を枝刈り対象にする

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.9,
)  # 大域的・非構造・強度枝刈り

model(input_image)  # ウォームアップ

start_time = time.time()
with torch.no_grad():
    output = model(input_image)
end_time = time.time()
print(f"推論時間（枝刈り直後）: {end_time - start_time:.4f} 秒")
# Intel Core i7-12700 で 0.0359 秒
# マスクを都度適用しているのでかえって遅くなる
'''
