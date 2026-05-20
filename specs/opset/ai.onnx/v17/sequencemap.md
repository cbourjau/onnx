---
op: SequenceMap
domain: ai.onnx
since_version: 17
deterministic: false
type_constraints:
- name: S
  doc: Constrain input types to any sequence type.
  allowed:
  - seq(tensor(bool))
  - seq(tensor(complex128))
  - seq(tensor(complex64))
  - seq(tensor(double))
  - seq(tensor(float))
  - seq(tensor(float16))
  - seq(tensor(int16))
  - seq(tensor(int32))
  - seq(tensor(int64))
  - seq(tensor(int8))
  - seq(tensor(string))
  - seq(tensor(uint16))
  - seq(tensor(uint32))
  - seq(tensor(uint64))
  - seq(tensor(uint8))
- name: V
  doc: Constrain to any tensor or sequence type.
  allowed:
  - seq(tensor(bool))
  - seq(tensor(complex128))
  - seq(tensor(complex64))
  - seq(tensor(double))
  - seq(tensor(float))
  - seq(tensor(float16))
  - seq(tensor(int16))
  - seq(tensor(int32))
  - seq(tensor(int64))
  - seq(tensor(int8))
  - seq(tensor(string))
  - seq(tensor(uint16))
  - seq(tensor(uint32))
  - seq(tensor(uint64))
  - seq(tensor(uint8))
  - tensor(bool)
  - tensor(complex128)
  - tensor(complex64)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(string)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: input_sequence
  type: S
  doc: Input sequence.
- name: additional_inputs
  type: V
  doc: Additional inputs to the graph
  kind:
    homogeneous: false
    min_arity: 0
outputs:
- name: out_sequence
  type: S
  doc: Output sequence(s)
  kind:
    homogeneous: false
    min_arity: 1
attributes:
- name: body
  type: graph
  doc: The graph to be run for each sample in the sequence(s). It should have as many
    inputs and outputs as inputs and outputs to the SequenceMap function.
  required: true
---

Applies a sub-graph to each sample in the input sequence(s).

Inputs can be either tensors or sequences, with the exception of the first input which must
be a sequence. The length of the first input sequence will determine the number of samples in the
outputs. Any other sequence inputs should have the same number of samples. The number of inputs
and outputs, should match the one of the subgraph.

For each i-th element in the output, a sample will be extracted from the input sequence(s) at
the i-th position and the sub-graph will be applied to it.
The outputs will contain the outputs of the sub-graph for each sample, in the same order as in
the input.

This operator assumes that processing each sample is independent and could executed in parallel
or in any order. Users cannot expect any specific ordering in which each subgraph is computed.
