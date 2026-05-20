---
inherit: v7/gru
since_version: 14
inputs:
  update:
  - name: X
    type: T
    doc: The input sequences packed (and potentially padded) into one 3-D tensor with
      the shape of `[seq_length, batch_size, input_size]`.
    differentiable: true
  - name: W
    type: T
    doc: The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]`
      (if bidirectional) along dimension 0. This tensor has shape `[num_directions,
      3*hidden_size, input_size]`.
    differentiable: true
  - name: R
    type: T
    doc: The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` (if
      bidirectional) along dimension 0. This tensor has shape `[num_directions, 3*hidden_size,
      hidden_size]`.
    differentiable: true
  - name: B
    type: T
    doc: 'The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and
      `[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor has
      shape `[num_directions, 6*hidden_size]`. Optional: If not specified - assumed
      to be 0'
    kind: Option
    differentiable: true
  - name: sequence_lens
    type: T1
    doc: Optional tensor specifying lengths of the sequences in a batch. If not specified
      - assumed all sequences in the batch to have length `seq_length`. It has shape
      `[batch_size]`.
    kind: Option
    differentiable: false
  - name: initial_h
    type: T
    doc: Optional initial value of the hidden. If not specified - assumed to be 0.
      It has shape `[num_directions, batch_size, hidden_size]`.
    kind: Option
    differentiable: false
outputs:
  update:
  - name: Y
    type: T
    doc: 'A tensor that concats all the intermediate output values of the hidden.
      It has shape `[seq_length, num_directions, batch_size, hidden_size]`. '
    kind: Option
    differentiable: true
  - name: Y_h
    type: T
    doc: The last output value of the hidden. It has shape `[num_directions, batch_size,
      hidden_size]`.
    kind: Option
    differentiable: true
attributes:
  add:
  - name: layout
    type: int
    doc: 'The shape format of inputs X, initial_h and outputs Y, Y_h. If 0, the following
      shapes are expected: X.shape = [seq_length, batch_size, input_size], Y.shape
      = [seq_length, num_directions, batch_size, hidden_size], initial_h.shape = Y_h.shape
      = [num_directions, batch_size, hidden_size]. If 1, the following shapes are
      expected: X.shape = [batch_size, seq_length, input_size], Y.shape = [batch_size,
      seq_length, num_directions, hidden_size], initial_h.shape = Y_h.shape = [batch_size,
      num_directions, hidden_size].'
    required: false
    default: 0
---

Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

* `X` - input tensor
* `z` - update gate
* `r` - reset gate
* `h` - hidden gate
* `t` - time step (t-1 means previous time step)
* `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
* `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
* `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
* `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
* `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
* `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
* `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
* `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE:
  Below are optional

* Affine(x)              - alpha * x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha * Tanh(beta * x)
* HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

* zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
* rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
* ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
* ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
* Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
