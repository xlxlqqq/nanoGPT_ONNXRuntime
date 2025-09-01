# 项目介绍
fork 自 https://github.com/karpathy/nanoGPT.git

# quick start
## 模型训练

模型训练有如下命令：
```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

## 模型转换
运行如下命令：
```bash
python trans2onnx.py
```

## 加速方案
### 量化
CPU上：FP16量化->int8量化：从7ms->4ms
GPU上：int8的推理效果不明显，原因是消费级的cuda core本来就不对int8做优化


## 模型推理
用onnx runtime 推理
版本：https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1




