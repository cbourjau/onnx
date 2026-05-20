---
op: TreeEnsembleRegressor
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T
  doc: The input type must be a tensor of a numeric type.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
inputs:
- name: X
  type: T
  doc: Input of shape [N,F]
outputs:
- name: Y
  type: tensor(float)
  doc: N classes
attributes:
- name: aggregate_function
  type: string
  doc: Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,'
    'SUM,' 'MIN,' 'MAX.'
  required: false
  default: SUM
- name: base_values
  type: floats
  doc: Base values for classification, added to final class score; the size must be
    the same as the classes or can be left unassigned (assumed 0)
  required: false
- name: n_targets
  type: int
  doc: The total number of targets.
  required: false
- name: nodes_falsenodeids
  type: ints
  doc: Child node if expression is false
  required: false
- name: nodes_featureids
  type: ints
  doc: Feature id for each node.
  required: false
- name: nodes_hitrates
  type: floats
  doc: Popularity of each node, used for performance and may be omitted.
  required: false
- name: nodes_missing_value_tracks_true
  type: ints
  doc: 'For each node, define what to do in the presence of a NaN: use the ''true''
    (if the attribute value is 1) or ''false'' (if the attribute value is 0) branch
    based on the value in this array.<br>This attribute may be left undefined and
    the default value is false (0) for all nodes.'
  required: false
- name: nodes_modes
  type: strings
  doc: The node kind, that is, the comparison to make at the node. There is no comparison
    to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT',
    'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
  required: false
- name: nodes_nodeids
  type: ints
  doc: Node id for each node. Node ids must restart at zero for each tree and increase
    sequentially.
  required: false
- name: nodes_treeids
  type: ints
  doc: Tree id for each node.
  required: false
- name: nodes_truenodeids
  type: ints
  doc: Child node if expression is true
  required: false
- name: nodes_values
  type: floats
  doc: Thresholds to do the splitting on for each node.
  required: false
- name: post_transform
  type: string
  doc: Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,'
    'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'
  required: false
  default: NONE
- name: target_ids
  type: ints
  doc: The index of the target that each weight is for
  required: false
- name: target_nodeids
  type: ints
  doc: The node id of each weight
  required: false
- name: target_treeids
  type: ints
  doc: The id of the tree that each node is in.
  required: false
- name: target_weights
  type: floats
  doc: The weight for each target
  required: false
---

Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
