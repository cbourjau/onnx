---
op: DictVectorizer
domain: ai.onnx.ml
since_version: 1
type_constraints:
- name: T1
  doc: The input must be a map from strings or integers to either strings or a numeric
    type. The key and value types cannot be the same.
  allowed:
  - map(int64, double)
  - map(int64, float)
  - map(int64, string)
  - map(string, double)
  - map(string, float)
  - map(string, int64)
- name: T2
  doc: The output will be a tensor of the value type of the input map. It's shape
    will be [1,C], where C is the length of the input dictionary.
  allowed:
  - tensor(double)
  - tensor(float)
  - tensor(int64)
  - tensor(string)
inputs:
- name: X
  type: T1
  doc: A dictionary.
outputs:
- name: Y
  type: T2
  doc: A 1-D tensor holding values from the input dictionary.
attributes:
- name: int64_vocabulary
  type: ints
  doc: An integer vocabulary array.<br>One and only one of the vocabularies must be
    defined.
  required: false
- name: string_vocabulary
  type: strings
  doc: A string vocabulary array.<br>One and only one of the vocabularies must be
    defined.
  required: false
---

Uses an index mapping to convert a dictionary to an array.<br>
    Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
    the key type. The index into the vocabulary array at which the key is found is then
    used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
    The key type of the input map must correspond to the element type of the defined vocabulary attribute.
    Therefore, the output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the output array.
    Any keys not present in the input dictionary, will be zero in the output array.<br>
    For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
