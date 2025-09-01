# 将模型从PT转换成ONNX模型的脚本

import torch
import torch.onnx
from model import GPT, GPTConfig
import json
import os

def load_checkpoint():
    # 加载模型配置
    checkpoint = torch.load('out-shakespeare-char/ckpt.pt')
    # 直接使用checkpoint中的模型参数
    model_args = checkpoint['model_args']
    
    # 创建模型实例
    # create a new model from the config in the checkpoint
    model_config = GPTConfig(**model_args)
    model = GPT(model_config)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    return model

def convert_to_onnx(model, output_path='model.onnx'):
    # 设置输入
    batch_size = 1
    sequence_length = model.config.block_size
    input_shape = (batch_size, sequence_length)
    
    # 创建示例输入
    dummy_input = torch.randint(0, model.config.vocab_size, input_shape)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=14,


        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print(f'Model exported to {output_path}')

def main():
    # 加载模型
    model = load_checkpoint()
    
    # 转换为ONNX
    convert_to_onnx(model)

if __name__ == '__main__':
    main()