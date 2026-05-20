---
inherit: v1/treeensembleregressor
since_version: 3
attributes:
  add:
  - name: base_values_as_tensor
    type: tensor
    doc: Base values for regression, added to final prediction after applying aggregate_function;
      the size must be the same as the classes or can be left unassigned (assumed
      0)
    required: false
  - name: nodes_hitrates_as_tensor
    type: tensor
    doc: Popularity of each node, used for performance and may be omitted.
    required: false
  - name: nodes_values_as_tensor
    type: tensor
    doc: Thresholds to do the splitting on for each node.
    required: false
  - name: target_weights_as_tensor
    type: tensor
    doc: The weight for each target
    required: false
  update:
  - name: base_values
    type: floats
    doc: Base values for regression, added to final prediction after applying aggregate_function;
      the size must be the same as the classes or can be left unassigned (assumed
      0)
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
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
