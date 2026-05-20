---
op: MelWeightMatrix
domain: ai.onnx
since_version: 17
type_constraints:
- name: T1
  doc: Constrain to integer tensors.
  allowed:
  - tensor(int32)
  - tensor(int64)
- name: T2
  doc: Constrain to float tensors
  allowed:
  - tensor(bfloat16)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: T3
  doc: Constrain to any numerical types.
  allowed:
  - tensor(bfloat16)
  - tensor(double)
  - tensor(float)
  - tensor(float16)
  - tensor(int16)
  - tensor(int32)
  - tensor(int64)
  - tensor(int8)
  - tensor(uint16)
  - tensor(uint32)
  - tensor(uint64)
  - tensor(uint8)
inputs:
- name: num_mel_bins
  type: T1
  doc: The number of bands in the mel spectrum.
  differentiable: false
- name: dft_length
  type: T1
  doc: The size of the original DFT. The size of the original DFT is used to infer
    the size of the onesided DFT, which is understood to be floor(dft_length/2) +
    1, i.e. the spectrogram only contains the nonredundant DFT bins.
  differentiable: false
- name: sample_rate
  type: T1
  doc: Samples per second of the input signal used to create the spectrogram. Used
    to figure out the frequencies corresponding to each spectrogram bin, which dictates
    how they are mapped into the mel scale.
  differentiable: false
- name: lower_edge_hertz
  type: T2
  doc: Lower bound on the frequencies to be included in the mel spectrum. This corresponds
    to the lower edge of the lowest triangular band.
  differentiable: false
- name: upper_edge_hertz
  type: T2
  doc: The desired top edge of the highest frequency band.
  differentiable: false
outputs:
- name: output
  type: T3
  doc: 'The Mel Weight Matrix. The output has the shape: [floor(dft_length/2) + 1][num_mel_bins].'
  differentiable: false
attributes:
- name: output_datatype
  type: int
  doc: 'The data type of the output tensor. Strictly must be one of the values from
    DataType enum in TensorProto whose values correspond to T3. The default value
    is 1 = FLOAT. '
  required: false
  default: 1
---

Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
