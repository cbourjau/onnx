---
op: GRU
domain: ai.onnx
since_version: 1
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T1
  doc: Constrain seq_lens to integer tensor.
  allowed:
  - tensor(int32)
inputs:
- name: X
  type: T
  doc: The input sequences packed (and potentially padded) into one 3-D tensor with
    the shape of `[seq_length, batch_size, input_size]`.
- name: W
  type: T
  doc: The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` (if
    bidirectional) along dimension 0. This tensor has shape `[num_directions, 3*hidden_size,
    input_size]`.
- name: R
  type: T
  doc: The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` (if bidirectional)
    along dimension 0. This tensor has shape `[num_directions, 3*hidden_size, hidden_size]`.
- name: B
  type: T
  doc: 'The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and `[WBb[zrh],
    RBb[zrh]]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions,
    6*hidden_size]`. Optional: If not specified - assumed to be 0'
  kind: Option
- name: sequence_lens
  type: T1
  doc: Optional tensor specifying lengths of the sequences in a batch. If not specified
    - assumed all sequences in the batch to have length `seq_length`. It has shape
    `[batch_size]`.
  kind: Option
- name: initial_h
  type: T
  doc: Optional initial value of the hidden. If not specified - assumed to be 0. It
    has shape `[num_directions, batch_size, hidden_size]`.
  kind: Option
outputs:
- name: Y
  type: T
  doc: A tensor that concats all the intermediate output values of the hidden. It
    has shape `[seq_length, num_directions, batch_size, hidden_size]`. It is optional
    if `output_sequence` is 0.
  kind: Option
- name: Y_h
  type: T
  doc: The last output value of the hidden. It has shape `[num_directions, batch_size,
    hidden_size]`.
attributes:
- name: activation_alpha
  type: floats
  doc: Optional scaling values used by some activation functions. The values are consumed
    in the order of activation functions, for example (f, g, h) in LSTM.
  required: false
- name: activation_beta
  type: floats
  doc: Optional scaling values used by some activation functions. The values are consumed
    in the order of activation functions, for example (f, g, h) in LSTM.
  required: false
- name: activations
  type: strings
  doc: 'A list of 2 (or 4 if bidirectional) activation functions for update, reset,
    and hidden gates. The activation functions must be one of the activation functions
    specified above. Optional: See the equations for default if not specified.'
  required: false
- name: clip
  type: float
  doc: Cell clip threshold. Clipping bounds the elements of a tensor in the range
    of [-threshold, +threshold] and is applied to the input of activations. No clip
    if not specified.
  required: false
- name: direction
  type: string
  doc: Specify if the RNN is forward, reverse, or bidirectional. Must be one of forward
    (default), reverse, or bidirectional.
  required: false
  default: foward
- name: hidden_size
  type: int
  doc: Number of neurons in the hidden layer
  required: false
- name: output_sequence
  type: int
  doc: The sequence output for the hidden is optional if 0. Default 0.
  required: false
  default: 0
---

Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
