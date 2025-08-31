// nano GPT的onnx runtime推理示例
// 参考：https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/CUDA-ExecutionProvider.md

#include <iostream>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <chrono>

int main() {
    try {
        // 1️⃣ 初始化全局环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "demo");

        // 2️⃣ 创建 SessionOptions
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 3️⃣ 模型路径
        const wchar_t* model_path = L"Q:\\xlxlqqq\\documents\\Projects\\GPT2\\nanoGPT\\model.onnx";

        // 4️⃣ 创建 Session
        Ort::Session session(env, model_path, session_options);

        std::cout << "ONNX Runtime session created successfully!" << std::endl;

        // 5️⃣ 准备输入张量
        // 假设模型输入是单个 int64 序列 [batch_size, sequence_length]
        std::vector<int64_t> input_ids = { 0, 1, 2 };  // 示例输入
        std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(input_ids.size()) };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

        // 6️⃣ 输入输出名称
        const char* input_names[] = { "input_ids" };
        const char* output_names[] = { "logits" };

        // 7️⃣ 执行推理

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<Ort::Value> output_tensors;
        for (int i = 0; i < 100; i++) {
            output_tensors = session.Run(
                Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Inference ran successfully!" << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Inference took " << duration / 100.0 << "ms" << std::endl;

        // 8️⃣ 获取输出数据
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i] << (i + 1 < output_shape.size() ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // 打印前几个输出数值
        std::cout << "Output values: ";
        for (size_t i = 0; i < std::min((size_t)10, output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()); ++i) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Std exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
