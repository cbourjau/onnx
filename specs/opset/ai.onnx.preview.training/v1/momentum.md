---
op: Momentum
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
  doc: Constrain input types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
inputs:
- name: R
  type: T1
  doc: The learning rate.
- name: T
  type: T2
  doc: Update count of "X". It should be a scalar.
- name: inputs
  type: T3
  doc: It sequentially contains the current values of optimized tensors, then their
    gradient tensors, and finally their momentum tensors. For example, if two tensors
    "X_1" and "X_2" are optimized, The expected input list would be ["X_1", "X_2",
    gradient of "X_1", gradient of "X_2", momentum of "X_1", momentum of "X_2"].
  kind:
    homogeneous: false
    min_arity: 1
outputs:
- name: outputs
  type: T3
  doc: It sequentially contains the new values of optimized tensors and then the new
    values of their momentum tensors. For example, if two tensors "X_1" and "X_2"
    are optimized, the output list would be [new value of "X_1," new value of "X_2"
    new momentum of "X_1", new momentum of "X_2"].
  kind:
    homogeneous: false
    min_arity: 1
attributes:
- name: alpha
  type: float
  doc: The decay factor of momentum. It should be a scalar.
  required: true
- name: beta
  type: float
  doc: The coefficient of gradient in computing new momentum. It should be a scalar.
  required: true
- name: mode
  type: string
  doc: Its value should be either "nesterov" or "standard". The value "nesterov" leads
    to the use of Nesterov's momentum while "standard" invokes stochastic gradient
    method using standard momentum
  required: true
- name: norm_coefficient
  type: float
  doc: Coefficient of 0.5 * norm_coefficient * ||X||^2.
  required: true
---

Compute one iteration of stochastic gradient update with momentum.
    This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, SG with momentum requires
    several parameters:

     - The learning-rate "R".
     - The update count "T". That is, the number of conducted training iterations. It should
       be zero in the first training iteration.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A decay coefficient of previous accumulated gradient (i.e., momentum) "alpha".
     - The scaling coefficient of current gradient "beta".
     - An attribute to choose either standard momentum or Nesterov's momentum "mode" should
       be used.

    For the sake of simplicity, assume that there is only one tensor (called "X") to be optimized.
    Other necessary inputs are "X"'s gradient (called "G") and "X"'s momentum (called "V"). This
    Momentum operator maps all these inputs to the new value of "X" (called "X_new") and its new
    momentum (called "V_new").

    This operator supports two different momentum algorithms. Set the attribute "mode" to
    "nesterov" if Nesterov's momentum is desired. Otherwise, set the attribute "model" to
    "standard" to use standard momentum. Computation details are described subsequently.

    Let "+", "-", "*", and "/" are all element-wise operations with numpy-style broadcasting.

    Pseudo code for SG with standard momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized

      // Update X.
      X_new = X - R * V_new

    Pseudo code for SG with Nesterov's momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G;

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized;

      // Compute final update direction and then update X.
      X_new = X - R * (G_regularized + alpha * V_new)

    If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2". The same
    pseudo code would be extended to handle all tensors jointly. More specifically, we can view "X" as a
    concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
    be concatenated too) and then our pseudo code becomes applicable.
