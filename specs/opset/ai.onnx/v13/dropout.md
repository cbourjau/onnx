---
inherit: v12/dropout
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to float tensors.
    allowed:
    - tensor(bfloat16)
    - tensor(double)
    - tensor(float)
    - tensor(float16)
inputs:
  update:
  - name: data
    type: T
    doc: The input data as Tensor.
    differentiable: true
  - name: ratio
    type: T1
    doc: The ratio of random dropout, with value in [0, 1). If this input was not
      set, or if it was set to 0, the output would be a simple copy of the input.
      If it's non-zero, output will be a random dropout of the scaled input, which
      is typically the case during training. It is an optional value, if not specified
      it will default to 0.5.
    kind: Option
    differentiable: false
  - name: training_mode
    type: T2
    doc: If set to true then it indicates dropout is being used for training. It is
      an optional value hence unless specified explicitly, it is false. If it is false,
      ratio is ignored and the operation mimics inference mode where nothing will
      be dropped from the input data and if mask is requested as output it will contain
      all ones.
    kind: Option
    differentiable: false
outputs:
  update:
  - name: output
    type: T
    doc: The output.
    differentiable: true
  - name: mask
    type: T2
    doc: The output mask.
    kind: Option
    differentiable: false
---
