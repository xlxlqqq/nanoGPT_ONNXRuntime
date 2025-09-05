cmake_minimum_required(VERSION 3.10)
project(custom_ops)

set(CMAKE_CXX_STANDARD 14)

# 下面两行为通用 fallback（把 ONNX Runtime include/lib 路径改成你的实际路径）
include_directories(Q:/xlxlqqq/Library/ONNXRuntime_cuda121/include)
link_directories(Q:/xlxlqqq/Library/ONNXRuntime_cuda121/lib)

add_library(custom_ops SHARED fused_gemm_customop.cc)
target_link_libraries(custom_ops onnxruntime) # 或者 onnxruntime_providers_shared / onnxruntime depending on your build
set_target_properties(custom_ops PROPERTIES PREFIX "lib" SUFFIX ".so")
