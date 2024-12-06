import torch
import torch.fx as fx
import torchvision

import fx_ir.converter.fx as fx_converter


def create_resnet50() -> fx.GraphModule:
    module = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    graph = fx.Tracer().trace(root=module)
    graph_module = fx.GraphModule(root=module, graph=graph)
    return graph_module


def print_opcodes(graph_module: fx.GraphModule) -> None:
    opcodes = fx_converter.find_src_opcodes(graph_module)
    for op in opcodes:
        print(op)


def main() -> None:
    src_graph_module = create_resnet50()
    print_opcodes(src_graph_module)
    x = torch.randn(1, 3, 224, 224)
    y = src_graph_module(x)
    tgt_graph_module = fx_converter.FXConverter(debug=False).convert(src_graph_module)
    inputs = (x,)
    outputs = tgt_graph_module(inputs)
    print(torch.equal(y, outputs[0]))


if __name__ == "__main__":
    main()
