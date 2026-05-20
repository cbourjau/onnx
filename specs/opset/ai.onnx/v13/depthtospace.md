---
inherit: v11/depthtospace
since_version: 13
type_constraints:
  update:
  - name: T
    doc: Constrain input and output types to all tensor types.
    allowed:
    - tensor(bfloat16)
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
  update:
  - name: input
    type: T
    doc: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or
      depth, H is the height and W is the width.
    differentiable: true
outputs:
  update:
  - name: output
    type: T
    doc: Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].
    differentiable: true
---

DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
```

In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
```
