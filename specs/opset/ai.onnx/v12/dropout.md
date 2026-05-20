---
inherit: v10/dropout
since_version: 12
type_constraints:
  add:
  - name: T2
    doc: Constrain output 'mask' types to boolean tensors.
    allowed:
    - tensor(bool)
  update:
  - name: T1
    doc: Constrain input 'ratio' types to float tensors.
    allowed:
    - tensor(double)
    - tensor(float)
    - tensor(float16)
inputs:
  add:
  - name: ratio
    type: T1
    doc: The ratio of random dropout, with value in [0, 1). If this input was not
      set, or if it was set to 0, the output would be a simple copy of the input.
      If it's non-zero, output will be a random dropout of the scaled input, which
      is typically the case during training. It is an optional value, if not specified
      it will default to 0.5.
    kind: Option
  - name: training_mode
    type: T2
    doc: If set to true then it indicates dropout is being used for training. It is
      an optional value hence unless specified explicitly, it is false. If it is false,
      ratio is ignored and the operation mimics inference mode where nothing will
      be dropped from the input data and if mask is requested as output it will contain
      all ones.
    kind: Option
outputs:
  update:
  - name: mask
    type: T2
    doc: The output mask.
    kind: Option
attributes:
  add:
  - name: seed
    type: int
    doc: (Optional) Seed to the random generator, if not specified we will auto generate
      one.
    required: false
  remove:
  - ratio
---

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
