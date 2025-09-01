import numpy as np
import onnxruntime as ort

def compare_models(int8_model_path, fp16_model_path, input_data):
    """
    比较FP32和FP16量化模型的输出差异
    
    参数:
        int8_model_path: 原始FP32模型路径
        fp16_model_path: 量化后的FP16模型路径
        input_data: 输入数据
    """
    # 创建两个模型的会话
    sess_fp32 = ort.InferenceSession(int8_model_path)
    sess_fp16 = ort.InferenceSession(fp16_model_path)
    
    # 准备输入
    input_name = sess_fp32.get_inputs()[0].name
    inputs = {input_name: input_data}
    
    # 运行推理
    out_fp32 = sess_fp32.run(None, inputs)
    out_fp16 = sess_fp16.run(None, inputs)
    
    # 计算差异
    diff = np.abs(out_fp32[0] - out_fp16[0])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"最大差异: {max_diff}")
    print(f"平均差异: {mean_diff}")
    
    return max_diff, mean_diff

if __name__ == "__main__":
    # 示例输入数据 - 应与main_cpp.cpp中的输入格式一致
    input_data = np.array([[0, 1, 2]], dtype=np.int64)
    
    # 模型路径
    int8_model = "./model/model_INT8.onnx"
    fp16_model = "./model/model_FP16.onnx"
    
    # 比较模型
    max_diff, mean_diff = compare_models(int8_model, fp16_model, input_data)