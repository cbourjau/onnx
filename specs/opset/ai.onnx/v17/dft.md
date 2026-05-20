---
op: DFT
domain: ai.onnx
since_version: 17
type_constraints:
- name: T1
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(bfloat16)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T2
  doc: Constrain scalar length types to int64_t.
  allowed:
  - tensor(int32)
  - tensor(int64)
inputs:
- name: input
  type: T1
  doc: 'For real input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1].
    For complex input, the following shape is expected: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2].
    The first dimension is the batch dimension. The following N dimensions correspond
    to the signal''s dimensions. The final dimension represents the real and imaginary
    parts of the value in that order.'
  differentiable: false
- name: dft_length
  type: T2
  doc: The length of the signal as a scalar. If greater than the axis dimension, the
    signal will be zero-padded up to dft_length. If less than the axis dimension,
    only the first dft_length values will be used as the signal. If not provided,
    the default dft_length = signal_dim_axis, except for the IRFFT case (onesided=1,
    inverse=1), in which case the default dft_length is 2 * (signal_dim_axis - 1).
    It's an optional value.
  kind: Option
  differentiable: false
outputs:
- name: output
  type: T1
  doc: 'The Fourier Transform of the input vector. For standard DFT (onesided=0),
    the output shape is: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]
    (complex), with signal_dim_axis = dft_length. For RFFT (onesided=1, inverse=0),
    the output shape is: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][2]
    (one-sided complex), with signal_dim_axis = floor(dft_length/2) + 1. For IRFFT
    (onesided=1, inverse=1), the output shape is: [batch_idx][signal_dim1][signal_dim2]...[signal_dimN][1]
    (real), where signal_dim_axis = dft_length.'
attributes:
- name: axis
  type: int
  doc: The axis on which to perform the DFT. By default this value is set to 1, which
    corresponds to the first dimension after the batch index. Negative value means
    counting dimensions from the back. Accepted range is $[-r, -2] \cup [0, r-2]$
    where `r = rank(input)`. The last dimension is for representing complex numbers
    and thus is an invalid axis.
  required: false
  default: 1
- name: inverse
  type: int
  doc: Whether to perform the inverse discrete fourier transform. By default this
    value is set to 0, which corresponds to false.
  required: false
  default: 0
- name: onesided
  type: int
  doc: If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are
    used or returned because the real-to-complex Fourier transform satisfies the conjugate
    symmetry, i.e., X[m, w] = X[m, n_fft-w]*. When onesided=1 and inverse=0 (forward
    DFT), only real input is supported and a one-sided complex spectrum is returned
    (RFFT). When onesided=1 and inverse=1 (inverse DFT), only complex input is supported
    and a full real signal is returned (IRFFT). When invoked with real or complex
    valued input, the default value is 0. Values can be 0 or 1.
  required: false
  default: 0
---

Computes the discrete Fourier transform of input.
