---
inherit: v1/treeensembleclassifier
since_version: 3
attributes:
  add:
  - name: base_values_as_tensor
    type: tensor
    doc: Base values for classification, added to final class score; the size must
      be the same as the classes or can be left unassigned (assumed 0)
    required: false
  - name: class_weights_as_tensor
    type: tensor
    doc: The weight for the class in class_id.
    required: false
  - name: nodes_hitrates_as_tensor
    type: tensor
    doc: Popularity of each node, used for performance and may be omitted.
    required: false
  - name: nodes_values_as_tensor
    type: tensor
    doc: Thresholds to do the splitting on for each node.
    required: false
---

Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
