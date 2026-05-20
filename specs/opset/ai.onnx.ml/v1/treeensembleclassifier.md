---
op: TreeEnsembleClassifier
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input type must be a tensor of a numeric type.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int32)
  - tensor(int64)
- name: T2
  doc: The output type will be a tensor of strings or integers, depending on which
    of the classlabels_* attributes is used.
  allowed:
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T1
  doc: Input of shape [N,F]
outputs:
- name: Y
  type: T2
  doc: N, Top class for each point
- name: Z
  type: tensor(float)
  doc: The class score for each class, for each point, a tensor of shape [N,E].
attributes:
- name: base_values
  type: floats
  doc: Base values for classification, added to final class score; the size must be
    the same as the classes or can be left unassigned (assumed 0)
  required: false
- name: class_ids
  type: ints
  doc: The index of the class list that each weight is for.
  required: false
- name: class_nodeids
  type: ints
  doc: node id that this weight is for.
  required: false
- name: class_treeids
  type: ints
  doc: The id of the tree that this node is in.
  required: false
- name: class_weights
  type: floats
  doc: The weight for the class in class_id.
  required: false
- name: classlabels_int64s
  type: ints
  doc: Class labels if using integer labels.<br>One and only one of the 'classlabels_*'
    attributes must be defined.
  required: false
- name: classlabels_strings
  type: strings
  doc: Class labels if using string labels.<br>One and only one of the 'classlabels_*'
    attributes must be defined.
  required: false
- name: nodes_falsenodeids
  type: ints
  doc: Child node if expression is false.
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
  doc: 'For each node, define what to do in the presence of a missing value: if a
    value is missing (NaN), use the ''true'' or ''false'' branch based on the value
    in this array.<br>This attribute may be left undefined, and the default value
    is false (0) for all nodes.'
  required: false
- name: nodes_modes
  type: strings
  doc: The node kind, that is, the comparison to make at the node. There is no comparison
    to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT',
    'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'
  required: false
- name: nodes_nodeids
  type: ints
  doc: Node id for each node. Ids may restart at zero for each tree, but it not required
    to.
  required: false
- name: nodes_treeids
  type: ints
  doc: Tree id for each node.
  required: false
- name: nodes_truenodeids
  type: ints
  doc: Child node if expression is true.
  required: false
- name: nodes_values
  type: floats
  doc: Thresholds to do the splitting on for each node.
  required: false
- name: post_transform
  type: string
  doc: Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,'
    'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'
  required: false
  default: NONE
---

Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
