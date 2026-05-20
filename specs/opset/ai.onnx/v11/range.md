---
op: Range
domain: ai.onnx
since_version: 11
deterministic: false
function_body: |-
  <
    domain: "",
    opset_import: ["" : 11]
  >
  Range (start, limit, delta) => (output)
  {
     sub_result = Sub (limit, start)
     sub_result_casted = Cast <to: int = 1> (sub_result)
     delta_casted = Cast <to: int = 1> (delta)
     div_result = Div (sub_result_casted, delta_casted)
     ceil_result = Ceil (div_result)
     ceil_result_relu = Relu (ceil_result)
     ceil_result_relu_int = Cast <to: int = 7> (ceil_result_relu)
     ceil_result_relu_bool = Cast <to: int = 9> (ceil_result_relu)
     variadic_output, output = Loop (ceil_result_relu_int, ceil_result_relu_bool, start) <body: graph = loop_body_attribute (int64 i, bool cond,  prev) => ( cond_out,  current,  range) {
        cond_out = Identity (cond)
        current = Add (prev, delta)
        range = Identity (prev)
     }>
  }
type_constraints:
- name: T
  doc: Constrain input types to common numeric type tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: start
  type: T
  doc: Scalar. First entry for the range of output values.
- name: limit
  type: T
  doc: Scalar. Exclusive upper limit for the range of output values.
- name: delta
  type: T
  doc: Scalar. Value to step by.
outputs:
- name: output
  type: T
  doc: A 1-D tensor with same type as the inputs containing generated range of values.
---

Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:

```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```

The pseudocode determining the contents of the output is shown below:

```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```

Example 1

```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```

Example 2

```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```
