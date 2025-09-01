import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# int8 quantize
def quantize_onnx_model_int8(input_model_path, output_model_path):
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QInt8,  # 量化权重为Int8
    )
    print(f"Quantized model saved to {output_model_path}")


if __name__ == "__main__":
    input_onnx_model = "./model/model_FP16.onnx"
    output_onnx_model = "./model/model_int8.onnx"
    quantize_onnx_model_int8(input_onnx_model, output_onnx_model)