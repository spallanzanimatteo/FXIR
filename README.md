# FX Intermediate Representation

The FX Intermediate Representation (FX IR) is a specialized dialect of PyTorch's [FX](https://pytorch.org/docs/stable/fx.html) intermediate representation.
FX IR **reduces the ambiguities inherent to the original intermediate representation, facilitating the application of compiler algorithms** such as static analysis and graph rewriting.

## How does the FX IR work?

FX IR enforces the following **invariants** on a `Graph`.
1. **Node opcodes**. Each node must have one of the following opcodes:
   * `"placeholder"`, representing input `Tensor`s;
   * `"output"`, representing output `Tensor`s;
   * `"call_module"`, representing individual `Tensor`s and operations.
2. **I/O nodes**. A `Graph` must contain exactly one `"placeholder"` and exactly one `"output"`:
   * the `"placeholder"` node represents the collection of model inputs;
   * the `"output"` node represents the collection of model outputs.
3. **`Array` nodes**. A `"call_module"` node representing an individual `Tensor`, with the following signature:
   * consume a tuple of `Tensor`s;
   * produce a single `Tensor`.
4. **Non-`Array` nodes**. A `"call_module"` node representing an operation, with the following signature:
     * consume one or more `Tensor`s;
     * produce a tuple of `Tensor`s.

Thanks to the invariants, users of the FX IR can make the following **assumptions**.
* Every operation is represented as a non-`Array` `"call_module"` node.
* Every `Graph` is bipartite:
  * the **array partition** contains all the `Array` `"call_module"` nodes;
  * the **operator partition** contains the `"placeholder"` node, non-`Array` `"call_module"` nodes, and the `"output"` node.
