---
inherit: v14/batchnormalization
since_version: 15
type_constraints:
  add:
  - name: T1
    doc: Constrain scale and bias types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
  - name: T2
    doc: Constrain mean and variance types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
  remove:
  - U
inputs:
  update:
  - name: scale
    type: T1
    doc: Scale tensor of shape (C).
    differentiable: true
  - name: B
    type: T1
    doc: Bias tensor of shape (C).
    differentiable: true
  - name: input_mean
    type: T2
    doc: running (training) or estimated (testing) mean tensor of shape (C).
    differentiable: true
  - name: input_var
    type: T2
    doc: running (training) or estimated (testing) variance tensor of shape (C).
    differentiable: true
outputs:
  update:
  - name: running_mean
    type: T2
    doc: The running mean after the BatchNormalization operator.
    kind: Option
    differentiable: false
  - name: running_var
    type: T2
    doc: The running variance after the BatchNormalization operator. This op uses
      the population size (N) for calculating variance, and not the sample size N-1.
    kind: Option
    differentiable: false
attributes:
  update:
  - name: training_mode
    type: int
    doc: If set to true, it indicates BatchNormalization is being used for training,
      and outputs 1 and 2 are to be computed.
    required: false
    default: 0
---

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
