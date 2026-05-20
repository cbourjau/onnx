---
op: Adagrad
domain: ai.onnx.preview.training
since_version: 1
type_constraints:
- name: T1
  doc: Constrain input types to float scalars.
  allowed:
  - tensor(double)
  - tensor(float)
- name: T2
  doc: Constrain input types to 64-bit integer scalars.
  allowed:
  - tensor(int64)
- name: T3
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
inputs:
- name: R
  type: T1
  doc: The initial learning rate.
- name: T
  type: T2
  doc: The update count of "X". It should be a scalar.
- name: inputs
  type: T3
  doc: The current values of optimized tensors, followed by their respective gradients,
    followed by their respective accumulated squared gradients.For example, if two
    tensor "X_1" and "X_2" are optimized, The input list would be ["X_1", "X_2", gradient
    of "X_1", gradient of "X_2", accumulated squared gradient of "X_1", accumulated
    squared gradient of "X_2"].
  kind:
    homogeneous: false
    min_arity: 1
outputs:
- name: outputs
  type: T3
  doc: Updated values of optimized tensors, followed by their updated values of accumulated
    squared gradients. For example, if two tensor "X_1" and "X_2" are optimized, the
    output list would be [new value of "X_1," new value of "X_2" new accumulated squared
    gradient of "X_1", new accumulated squared gradient of "X_2"].
  kind:
    homogeneous: false
    min_arity: 1
attributes:
- name: decay_factor
  type: float
  doc: The decay factor of learning rate after one update.The effective learning rate
    is computed by r = R / (1 + T * decay_factor). Default to 0 so that increasing
    update counts doesn't reduce the learning rate.
  required: false
  default: 0.0
- name: epsilon
  type: float
  doc: Small scalar to avoid dividing by zero.
  required: false
  default: 9.999999974752427e-07
- name: norm_coefficient
  type: float
  doc: Regularization coefficient in 0.5 * norm_coefficient * ||X||_2^2. Default to
    0, which means no regularization.
  required: false
  default: 0.0
---

Compute one iteration of ADAGRAD, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    some parameters:

     - The initial learning-rate "R".
     - The update count "T". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A learning-rate decay factor "decay_factor".
     - A small constant "epsilon" to avoid dividing-by-zero.

    At each ADAGRAD iteration, the optimized tensors are moved along a direction
    computed based on their estimated gradient and accumulated squared gradient. Assume
    that only a single tensor "X" is updated by this operator. We need the value of "X",
    its gradient "G", and its accumulated squared gradient "H". Therefore, variables in
    this operator's input list are sequentially "R", "T", "X", "G", and "H". Other
    parameters are given as attributes because they are usually constants. Also, the
    corresponding output tensors are the new value of "X" (called "X_new"), and then
    the new accumulated squared gradient (called "H_new"). Those outputs are computed
    from the given inputs following the pseudo code below.

    Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Compute a scalar learning-rate factor. At the first update of X, T is generally
      // 0 (0-based update index) or 1 (1-based update index).
      r = R / (1 + T * decay_factor);

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G;

      // Compute new accumulated squared gradient.
      H_new = H + G_regularized * G_regularized;

      // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
      // computes element-wise square-root.
      H_adaptive = Sqrt(H_new) + epsilon

      // Compute the new value of "X".
      X_new = X - r * G_regularized / H_adaptive;

    If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2", the same
    pseudo code may be extended to handle all tensors jointly. More specifically, we can view "X" as a
    concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
    be concatenated too) and then just reuse the entire pseudo code.

    Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    In that reference paper, this operator is a special case of the Figure 1's composite mirror
    descent update.
