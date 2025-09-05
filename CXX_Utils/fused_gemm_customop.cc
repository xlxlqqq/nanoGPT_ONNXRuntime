#include "onnxruntime_c_api.h"
#include <vector>
#include <cstring>
#include <cmath>

// 获取 tensor 形状
struct TensorDims {
    std::vector<int64_t> dims;
    TensorDims(const OrtApi* api, const OrtValue* value) {
        OrtTensorTypeAndShapeInfo* info;
        api->GetTensorShapeAndType(value, &info);
        size_t dim_count;
        api->GetDimensionsCount(info, &dim_count);
        dims.resize(dim_count);
        api->GetDimensions(info, dims.data(), dim_count);
        api->ReleaseTensorTypeAndShapeInfo(info);
    }
    int64_t operator[](size_t i) const { return dims[i]; }
};

// 自定义 Kernel 结构体
struct FusedGemmKernel {
    const OrtApi* api;
    FusedGemmKernel(const OrtApi* api_, const OrtKernelInfo* info) : api(api_) {}
};

// Kernel Compute 函数
OrtStatus* FusedGemmKernel_Compute(void* op_kernel, OrtKernelContext* context) {
    FusedGemmKernel* kernel = (FusedGemmKernel*)op_kernel;
    const OrtApi* api = kernel->api;

    const OrtValue* A_value = nullptr;
    const OrtValue* B_value = nullptr;
    const OrtValue* C_value = nullptr;

    // 获取输入
    api->KernelContext_GetInput(context, 0, &A_value);
    api->KernelContext_GetInput(context, 1, &B_value);
    api->KernelContext_GetInput(context, 2, &C_value);

    // 获取数据指针
    float* A_data = nullptr;
    float* B_data = nullptr;
    float* C_data = nullptr;

    api->GetTensorMutableData(A_value, (void**)&A_data);
    api->GetTensorMutableData(B_value, (void**)&B_data);
    api->GetTensorMutableData(C_value, (void**)&C_data);

    // 获取 shape
    TensorDims dimA(api, A_value); // [M,K]
    TensorDims dimB(api, B_value); // [K,N]

    int64_t M = dimA[0];
    int64_t K = dimA[1];
    int64_t N = dimB[1];

    // 创建输出
    std::vector<int64_t> Y_dims = { M, N };
    OrtValue* Y_value = nullptr;
    api->KernelContext_GetOutput(context, 0, Y_dims.data(), Y_dims.size(), &Y_value);

    float* Y_data = nullptr;
    api->GetTensorMutableData(Y_value, (void**)&Y_data);

    // MatMul + Add (bias)
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A_data[m * K + k] * B_data[k * N + n];
            }
            Y_data[m * N + n] = sum + C_data[n]; // bias 按列加
        }
    }

    return nullptr; // 成功
}

// Kernel Destroy
void FusedGemmKernel_Destroy(void* op_kernel) {
    delete (FusedGemmKernel*)op_kernel;
}

// Kernel Create
void* FusedGemmKernel_Create(const OrtCustomOp* /*op*/, const OrtApi* api, const OrtKernelInfo* info) {
    return new FusedGemmKernel(api, info);
}

// Custom Op 定义
struct FusedGemmCustomOp {
    OrtCustomOp base;

    FusedGemmCustomOp() {
        base.version = ORT_API_VERSION;
        base.CreateKernel = FusedGemmKernel_Create;
        base.KernelCompute = FusedGemmKernel_Compute;
        base.KernelDestroy = FusedGemmKernel_Destroy;

        base.GetName = [](const OrtCustomOp* /*op*/) -> const char* { return "FusedGemm"; };
        base.GetExecutionProviderType = [](const OrtCustomOp* /*op*/) -> const char* { return nullptr; };

        base.GetInputTypeCount = [](const OrtCustomOp* /*op*/) -> size_t { return 3; };
        base.GetOutputTypeCount = [](const OrtCustomOp* /*op*/) -> size_t { return 1; };

        base.GetInputType = [](const OrtCustomOp* /*op*/, size_t index) -> ONNXTensorElementDataType {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        };
        base.GetOutputType = [](const OrtCustomOp* /*op*/, size_t index) -> ONNXTensorElementDataType {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        };
    }
};

// 全局 CustomOp 对象
static FusedGemmCustomOp c_CustomOp;

// 注册函数
extern "C" OrtStatus* ORT_API_CALL OrtRegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    api->CustomOpDomain_Add(nullptr, (OrtCustomOp*)&c_CustomOp.base);
    return nullptr;
}
