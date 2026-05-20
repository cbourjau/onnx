---
op: Adam
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
  doc: The tensors to be optimized, followed by their respective gradients, followed
    by their respective accumulated gradients (aka momentum), followed by their respective
    accumulated squared gradients. For example, to optimize tensors "X_1" and "X_2,",
    the input list would be ["X_1", "X_2", gradient of "X_1", gradient of "X_2", accumulated
    gradient of "X_1", accumulated gradient of "X_2", accumulated squared gradient
    of "X_1", accumulated squared gradient of "X_2"].
  kind:
    homogeneous: false
    min_arity: 1
outputs:
- name: outputs
  type: T3
  doc: New values of optimized tensors, followed by their respective new accumulated
    gradients, followed by their respective new accumulated squared gradients. For
    example, if two tensors "X_1" and "X_2" are optimized, the outputs list would
    be [new value of "X_1", new value of "X_2", new accumulated gradient of "X_1",
    new accumulated gradient of "X_2", new accumulated squared gradient of "X_1",
    new accumulated squared gradient of "X_2"].
  kind:
    homogeneous: false
    min_arity: 1
attributes:
- name: alpha
  type: float
  doc: Coefficient of previously accumulated gradient in running average. Default
    to 0.9.
  required: false
  default: 0.8999999761581421
- name: beta
  type: float
  doc: Coefficient of previously accumulated squared-gradient in running average.
    Default to 0.999.
  required: false
  default: 0.9990000128746033
- name: epsilon
  type: float
  doc: Small scalar to avoid dividing by zero.
  required: false
  default: 9.999999974752427e-07
- name: norm_coefficient
  type: float
  doc: Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to
    0, which means no regularization.
  required: false
  default: 0.0
- name: norm_coefficient_post
  type: float
  doc: Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to
    0, which means no regularization.
  required: false
  default: 0.0
---

Compute one iteration of Adam, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. First of all, Adam requires
    some parameters:

     - The learning-rate "R".
     - The update count "T". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A small constant "epsilon" to avoid dividing-by-zero.
     - Two coefficients, "alpha" and "beta".

    At each Adam iteration, the optimized tensors are moved along a direction
    computed based on their exponentially-averaged historical gradient and
    exponentially-averaged historical squared gradient. Assume that only a tensor
    "X" is being optimized. The rest of required information is

     - the value of "X",
     - "X"'s gradient (denoted by "G"),
     - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
     - "X"'s exponentially-averaged historical squared gradient (denoted by "H").

    Some of those parameters are passed into this operator as input tensors and others
    are stored as this operator's attributes. Specifically, this operator's input tensor
    list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
    the second input, and so on. Other parameters are given as attributes because they
    are constants. Moreover, the corresponding output tensors are

     - the new value of "X" (called "X_new"),
     - the new exponentially-averaged historical gradient (denoted by "V_new"), and
     - the new exponentially-averaged historical squared gradient (denoted by "H_new").

    Those outputs are computed following the pseudo code below.

    Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G

      // Update exponentially-averaged historical gradient.
      V_new = alpha * V + (1 - alpha) * G_regularized

      // Update exponentially-averaged historical squared gradient.
      H_new = beta * H + (1 - beta) * G_regularized * G_regularized

      // Compute the element-wise square-root of H_new. V_new will be element-wisely
      // divided by H_sqrt for a better update direction.
      H_sqrt = Sqrt(H_new) + epsilon

      // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
      R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

      // Compute new value of "X".
      X_new = X - R_adjusted * V_new / H_sqrt

      // Post-update regularization.
      X_final = (1 - norm_coefficient_post) * X_new

    If there are multiple inputs to be optimized, the pseudo code will be applied
    independently to each of them.
