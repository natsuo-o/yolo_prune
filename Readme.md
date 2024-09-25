## ultralyticsを用いたYolo枝刈り

1. `/opt/conda/lib/python3.10/site-packages/ultralytics/nn/task.py`、`/opt/conda/lib/python3.10/site-packages/ultralytics/engine/model.py`、`/opt/conda/lib/python3.10/site-packages/ultralytics/utils/torch_utils.py`のコードを修正する
    -  `/opt/conda/lib/python3.10/site-packages/ultralytics/nn/task.py`の修正

        203行目の`m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv`の１つ上の行に以下を追加
        ```
        m.conv.weight = m.conv.weight.to('cuda', dtype=torch.float32)

        ```
    - `/opt/conda/lib/python3.10/site-packages/ultralytics/engine/model.py`のコードを修正
        802行目の`self.trainer.hub_session = self.session`の１つ上の行に以下を追加
        ```
            import torch.nn.utils.prune as prune
            parameters_to_prune = [
            (module, "weight") for module in self.model.modules() if isinstance(module, torch.nn.Conv2d)
            ]  # すべての畳み込み層を枝刈り対象にする


            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.8,
            )  # 大域的・非構造・強度枝刈り
            self.model.to('cuda')
        ```
    - `/opt/conda/lib/python3.10/site-packages/ultralytics/utils/torch_utils.py`のコードを修正
        - 510行目の`self.ema = deepcopy(de_parallel(model)).eval()`をコメントアウトする
        - 509行目の`"""Initialize EMA for 'model' with given arguments."""`の１つ下の行に以下を追加する
        ```
        import torch.nn.utils.prune as prune

        new_model = deepcopy(type(model)())

        parameters_to_prune = [
            (module, "weight") for module in new_model.modules() if isinstance(module, torch.nn.Conv2d)
            ]  # すべての畳み込み層を枝刈り対象にす

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.0,
        )  # 大域的・非構造・強度枝刈り
        # metrics = model.val(data="coco8.yaml", device='cpu')

        new_model.load_state_dict(model.state_dict())
        self.ema = new_model.eval().to('cuda')
        ```
2. 上記の設定をすれば基本的に枝刈りをすることができる
    - 枝刈りしたモデルの学習は`train.py`で行う
    - 学習されたモデルは`/workspace/runs/detect/train/weights/best.pt`のような形式で保存される
    - `model.export(format='onnx') `でonnx形式に保存される。保存されたモデルは`/workspace/runs/detect/train/weights/best.onnx`に保存される

    - _＊枝刈りなしの普通の学習をする場合は、1.で変更した部分を元に戻す必要がある_
3. deepsparseを用いた推論
    - `python3 deepsparse_inference.py`を実行することでできる
    - `model_path`に2.で保存したonnx形式のモデルのパスを入力する
    