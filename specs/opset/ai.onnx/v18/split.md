---
inherit: v13/split
since_version: 18
attributes:
  add:
  - name: num_outputs
    type: int
    doc: Number of outputs to split parts of the tensor into. If the tensor is not
      evenly splittable the last chunk will be smaller.
    required: false
---

Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.
