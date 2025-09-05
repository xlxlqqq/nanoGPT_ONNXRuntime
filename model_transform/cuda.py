import onnx
from onnx import helper, numpy_helper

def fuse_matmul_add(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    # 构建一个节点输出到节点的映射，方便查找 MatMul -> Add
    output_to_node = {node.output[0]: node for node in graph.node}

    new_nodes = []
    skip_nodes = set()  # 跳过已经融合的 Add 节点

    for node in graph.node:
        if node.name in skip_nodes:
            continue

        # 只处理 MatMul
        if node.op_type == "MatMul":
            matmul_node = node
            matmul_output = matmul_node.output[0]

            # 检查下一个节点是否是 Add 且输入是这个 MatMul 输出
            add_node = None
            for candidate in graph.node:
                if candidate.op_type == "Add" and matmul_output in candidate.input:
                    add_node = candidate
                    break

            if add_node:
                # 创建 FusedGemm 节点
                # inputs: MatMul的两个输入 + Add 的另一个输入（bias）
                bias_input = [i for i in add_node.input if i != matmul_output][0]
                fused_node = helper.make_node(
                    "FusedGemm",
                    inputs=[matmul_node.input[0], matmul_node.input[1], bias_input],
                    outputs=[add_node.output[0]],
                    name=matmul_node.name + "_fusedgemm"
                )
                new_nodes.append(fused_node)
                skip_nodes.add(add_node.name)  # 跳过已经融合的 Add
                continue

        # 其他节点直接加入
        new_nodes.append(node)

    # 替换原来的节点
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    onnx.save(model, output_path)
    print(f"Fused MatMul->Add nodes saved to {output_path}")


if __name__ == "__main__":
    fuse_matmul_add("./model/model_FP16.onnx", "./model/nanoGPT_fused.onnx")
