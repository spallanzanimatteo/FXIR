import torch
import torch.fx as fx
import torch.nn as nn

import fx_ir.converter.fx.converter.infer_type as infer_type


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = torch.split(x, [2, 6], dim=0)
        y = torch.cat([x1, x0], dim=0)
        return y


def main() -> None:
    # create the model
    module = Model()
    graph = fx.Tracer().trace(root=module)
    graph_module = fx.GraphModule(root=module, graph=graph)
    # perform type inference
    infer_type.infer_type(graph_module, {"x": torch.Tensor(list(range(8)))})
    for node in graph_module.graph.nodes:
        print(node.op, node.name)
        print("\t", node.type)


if __name__ == "__main__":
    main()
