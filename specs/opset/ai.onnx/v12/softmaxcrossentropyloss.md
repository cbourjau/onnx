---
op: SoftmaxCrossEntropyLoss
domain: ai.onnx
since_version: 12
type_constraints:
- name: T
  doc: Constrain input and output types to float tensors.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(float16)
- name: Tind
  doc: Constrain target to integer types
  allowed:
  - tensor(int32)
  - tensor(int64)
inputs:
- name: scores
  type: T
  doc: The predicted outputs with shape [batch_size, class_size], or [batch_size,
    class_size, D1, D2 , ..., Dk], where K is the number of dimensions.
- name: labels
  type: Tind
  doc: The ground truth output tensor, with shape [batch_size], or [batch_size, D1,
    D2, ..., Dk], where K is the number of dimensions. Labels element value shall
    be in range of [0, C). If ignore_index is specified, it may have a value outside
    [0, C) and the label values should either be in the range [0, C) or have the value
    ignore_index.
- name: weights
  type: T
  doc: A manual rescaling weight given to each class. If given, it has to be a 1D
    Tensor assigning weight to each of the classes. Otherwise, it is treated as if
    having all ones.
  kind: Option
outputs:
- name: output
  type: T
  doc: Weighted loss float Tensor. If reduction is 'none', this has the shape of [batch_size],
    or [batch_size, D1, D2, ..., Dk] in case of K-dimensional loss. Otherwise, it
    is a scalar.
- name: log_prob
  type: T
  doc: Log probability tensor. If the output of softmax is prob, its value is log(prob).
  kind: Option
attributes:
- name: ignore_index
  type: int
  doc: Specifies a target value that is ignored and does not contribute to the input
    gradient. It's an optional value.
  required: false
- name: reduction
  type: string
  doc: 'Type of reduction to apply to loss: none, sum, mean(default). ''none'': no
    reduction will be applied, ''sum'': the output will be summed. ''mean'': the sum
    of the output will be divided by the number of elements in the output.'
  required: false
  default: mean
---

Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
