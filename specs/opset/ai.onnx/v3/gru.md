---
inherit: v1/gru
since_version: 3
outputs:
  update:
  - name: Y_h
    type: T
    doc: The last output value of the hidden. It has shape `[num_directions, batch_size,
      hidden_size]`.
    kind: Option
attributes:
  add:
  - name: linear_before_reset
    type: int
    doc: When computing the output of the hidden gate, apply the linear transformation
      before multiplying by the output of the reset gate.
    required: false
    default: 0
  update:
  - name: activation_alpha
    type: floats
    doc: Optional scaling values used by some activation functions. The values are
      consumed in the order of activation functions, for example (f, g, h) in LSTM.
      Default values are the same as of corresponding ONNX operators.For example with
      LeakyRelu, the default alpha is 0.01.
    required: false
  - name: activation_beta
    type: floats
    doc: Optional scaling values used by some activation functions. The values are
      consumed in the order of activation functions, for example (f, g, h) in LSTM.
      Default values are the same as of corresponding ONNX operators.
    required: false
  - name: direction
    type: string
    doc: Specify if the RNN is forward, reverse, or bidirectional. Must be one of
      forward (default), reverse, or bidirectional.
    required: false
    default: forward
---
