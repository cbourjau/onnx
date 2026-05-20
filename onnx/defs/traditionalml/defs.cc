// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/traditionalml/utils.h"
#include "onnx/defs/type_builders.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {
static constexpr const char* ArrayFeatureExtractor_ver1_doc = R"DOC(
    Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    ArrayFeatureExtractor,
    1,
    OpSchema().FillUsing(ArrayFeatureExtractor_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      const auto input_ndim = input_shape.dim_size();
      if (input_ndim == 1) {
        return;
      }
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      // This operator only applies to the last dimension; thus -1
      for (int i = 0; i < input_ndim - 1; ++i) {
        *output_shape->add_dim() = input_shape.dim(i);
      }

      // value of the output's last dimension is the total amount of indices
      // set Unknown length for the last dimension if it cannot be calculated
      auto last_dim = output_shape->add_dim();
      if (hasInputShape(ctx, 1)) {
        const auto& indices_shape = getInputShape(ctx, 1);
        if (indices_shape.dim_size() > 0) {
          int64_t num_indices = 1;
          std::string single_symbolic_dim;
          for (int i = 0; i < indices_shape.dim_size(); i++) {
            if (indices_shape.dim(i).has_dim_value()) {
              if (checked_mul_overflow(num_indices, indices_shape.dim(i).dim_value(), &num_indices)) {
                fail_shape_inference("Dimension product overflow in ArrayFeatureExtractor");
              }
            } else if (indices_shape.dim(i).has_dim_param()) {
              if (single_symbolic_dim.empty()) {
                // it is possible to set symbolic dimension param if the rest dim values are all
                // value 1
                single_symbolic_dim = indices_shape.dim(i).dim_param();
              } else {
                return;
              }
            } else {
              return;
            }
          }
          if (single_symbolic_dim.empty()) {
            last_dim->set_dim_value(num_indices);
          } else if (num_indices == 1) {
            last_dim->set_dim_param(single_symbolic_dim);
          }
        }
      }
    }));

static constexpr const char* Binarizer_ver1_doc = R"DOC(
    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Binarizer,
    1,
    OpSchema().FillUsing(Binarizer_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

static constexpr const char* CastMap_ver1_doc = R"DOC(
    Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    CastMap,
    1,
    OpSchema().FillUsing(CastMap_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto cast_to_attr = ctx.getAttribute("cast_to");
      auto output_type = ctx.getOutputType(0)->mutable_tensor_type();
      if (nullptr == cast_to_attr) {
        output_type->set_elem_type(TensorProto::FLOAT);
        return;
      }
      auto& cast_to = cast_to_attr->s();
      if ("TO_FLOAT" == cast_to) {
        output_type->set_elem_type(TensorProto::FLOAT);
      } else if ("TO_INT64" == cast_to) {
        output_type->set_elem_type(TensorProto::INT64);
      } else if ("TO_STRING" == cast_to) {
        output_type->set_elem_type(TensorProto::STRING);
      }
    }));

static constexpr const char* CategoryMapper_ver1_doc = R"DOC(
    Converts strings to integers and vice versa.<br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    CategoryMapper,
    1,
    OpSchema().FillUsing(CategoryMapper_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto cats_int64s = ctx.getAttribute("cats_int64s");
      if (cats_int64s == nullptr) {
        fail_shape_inference("Attribute 'cats_int64s' is required.");
      }
      auto cats_strings = ctx.getAttribute("cats_strings");
      if (cats_strings == nullptr) {
        fail_shape_inference("Attribute 'cats_strings' is required.");
      }
      if (cats_int64s->ints_size() != cats_strings->strings_size()) {
        fail_shape_inference("Attributes 'cats_int64s' and 'cats_strings' are required to be the same length.");
      }
      if (nullptr == ctx.getInputType(0)) {
        return;
      }
      auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
      if (TensorProto::STRING == input_elem_type) {
        updateOutputElemType(ctx, 0, TensorProto::INT64);
      } else if (TensorProto::INT64 == input_elem_type) {
        updateOutputElemType(ctx, 0, TensorProto::STRING);
      }
      if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

static constexpr const char* DictVectorizer_ver1_doc = R"DOC(
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
    )DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    DictVectorizer,
    1,
    OpSchema().FillUsing(DictVectorizer_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto input_elem_type = ctx.getInputType(0)->map_type().value_type().tensor_type().elem_type();
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      output_elem_type->set_elem_type(input_elem_type);
    }));

static constexpr const char* FeatureVectorizer_ver1_doc = R"DOC(
    Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimension. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(FeatureVectorizer, 1, OpSchema().FillUsing(FeatureVectorizer_v1_FillSpec));

static constexpr const char* Imputer_ver1_doc = R"DOC(
    Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    This operator is typically used to replace missing values in situations where they have a canonical
    representation, such as -1, 0, NaN, or some extreme value.<br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(Imputer, 1, OpSchema().FillUsing(Imputer_v1_FillSpec));

static constexpr const char* LabelEncoder_ver4_doc = R"DOC(
    Maps each element in the input tensor to another value.<br>
    The mapping is determined by the two parallel attributes, 'keys_*' and
    'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    would be mapped to the i-th value in the specified 'values_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys_*' should be identical while the output type is identical to the
    specified 'values_*' attribute. Note that the 'keys_*' and 'values_*' attributes
    must have the same length. If an input element can not be found in the
    specified 'keys_*' attribute, the 'default_*' that matches the specified
    'values_*' attribute may be used as its output value. The type of the 'default_*'
    attribute must match the 'values_*' attribute chosen. <br>
    Let's consider an example which maps a string tensor to an integer tensor.
    Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
    and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
    "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
    Since this operator is an one-to-one mapping, its input and output shapes
    are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    Float keys with value 'NaN' match any input 'NaN' value regardless of bit
    value. If a key is repeated, the last key takes precedence.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    4,
    OpSchema().FillUsing(LabelEncoder_v4_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto [key_type, key_length] =
          getAttributeElementTypeAndLength(ctx, {"keys_tensor", "keys_strings", "keys_int64s", "keys_floats"});
      if (key_type == TensorProto::UNDEFINED) {
        fail_shape_inference("At least one of keys_tensor, keys_strings, keys_int64s, keys_floats must be set.");
      }
      if (key_type != ctx.getInputType(0)->tensor_type().elem_type()) {
        fail_shape_inference(
            "The input type was ",
            ctx.getInputType(0)->tensor_type().elem_type(),
            " and the key type ",
            key_type,
            " are different, which is not permitted for LabelEncoders.");
      }

      auto [value_type, value_length] =
          getAttributeElementTypeAndLength(ctx, {"values_tensor", "values_strings", "values_int64s", "values_floats"});
      if (value_type == TensorProto::UNDEFINED) {
        fail_shape_inference(
            "At least one of values_tensor, values_strings, values_int64s, values_floats must be set.");
      }
      if (value_length != key_length) {
        fail_shape_inference(
            "The number of keys ",
            key_length,
            " and the number of values ",
            value_length,
            " must be the same in the LabelEncoder.");
      }

      auto default_attr = ctx.getAttribute("default_tensor");
      if (nullptr != default_attr && default_attr->has_t() && default_attr->t().has_data_type() &&
          default_attr->t().data_type() != TensorProto_DataType_UNDEFINED) {
        auto default_tensor = default_attr->t();
        if (default_tensor.data_type() != value_type) {
          fail_shape_inference(
              "The default tensor type ",
              default_tensor.data_type(),
              " and the value type ",
              value_type,
              " must be the same in the LabelEncoder.");
        }
        if (1 != default_tensor.dims_size() || 1 != default_tensor.dims(0)) {
          fail_shape_inference("The default tensor must be a singleton 1D tensor.");
        }
      }
      // Propagate shape from input type and assign output type based on value type
      ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(value_type);
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

static constexpr const char* LinearClassifier_ver1_doc = R"DOC(
    Linear classifier
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearClassifier,
    1,
    OpSchema().FillUsing(LinearClassifier_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      std::vector<std::string> label_strs;
      std::vector<int64_t> label_ints;

      auto labels_strings_present = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
      bool using_strings = (labels_strings_present && !label_strs.empty());

      if (!using_strings) {
        getRepeatedAttribute(ctx, "classlabels_ints", label_ints);
      }

      // Type inference
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      if (using_strings) {
        output_elem_type->set_elem_type(TensorProto::STRING);
      } else {
        output_elem_type->set_elem_type(TensorProto::INT64);
      }

      // second output is always of float type
      ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

      // Shape/Rank inference begins

      // establish the number of classes
      std::vector<float> intercepts;
      getRepeatedAttribute(ctx, "intercepts", intercepts);
      int class_count = static_cast<int>(intercepts.size());
      if (intercepts.size() == 1 &&
          ((using_strings && label_strs.size() == 2) || (!using_strings && label_ints.size() == 2))) {
        class_count = 2;
      }

      TensorShapeProto_Dimension batch_size_dim, class_count_dim;
      class_count_dim.set_dim_value(class_count);

      if (hasNInputShapes(ctx, 1)) {
        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        const auto input_rank = input_shape.dim_size();
        if (input_rank == 1) {
          // if input_rank is 1, batch_size is interpreted to be 1
          batch_size_dim.set_dim_value(1);
        } else if (input_rank == 2) {
          batch_size_dim = input_shape.dim(0);
        } else {
          fail_shape_inference("Input's shape should be 1D or 2D");
        }
      }

      updateOutputShape(ctx, 0, {batch_size_dim});
      updateOutputShape(ctx, 1, {batch_size_dim, class_count_dim});
    }));

static constexpr const char* LinearRegressor_ver1_doc = R"DOC(
    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(LinearRegressor, 1, OpSchema().FillUsing(LinearRegressor_v1_FillSpec));

static constexpr const char* Normalizer_ver1_doc = R"DOC(
    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(Normalizer, 1, OpSchema().FillUsing(Normalizer_v1_FillSpec));

static constexpr const char* OneHotEncoder_ver1_doc = R"DOC(
    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8,
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    OneHotEncoder,
    1,
    OpSchema().FillUsing(OneHotEncoder_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      std::vector<int64_t> cats_int64s;
      bool has_int64s = getRepeatedAttribute(ctx, "cats_int64s", cats_int64s);
      std::vector<std::string> cats_strings;
      bool has_strings = getRepeatedAttribute(ctx, "cats_strings", cats_strings);
      if (has_int64s == has_strings) {
        fail_shape_inference("Exactly one of 'cats_*' attributes must be provided.");
      }
      // Check if input shape is available before accessing it
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      for (int i = 0; i < input_shape.dim_size(); i++) {
        *shape->add_dim() = input_shape.dim(i);
      }
      shape->add_dim()->set_dim_value(std::max(cats_int64s.size(), cats_strings.size()));
      updateOutputElemType(ctx, 0, TensorProto::FLOAT);
    }));

static constexpr const char* Scaler_ver1_doc = R"DOC(
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(Scaler, 1, OpSchema().FillUsing(Scaler_v1_FillSpec));

static constexpr const char* SVMClassifier_ver1_doc = R"DOC(
    Support Vector Machine classifier
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMClassifier,
    1,
    OpSchema().FillUsing(SVMClassifier_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      std::vector<std::string> label_strs;
      auto result = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
      bool using_strings = (result && !label_strs.empty());
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      if (using_strings) {
        output_elem_type->set_elem_type(TensorProto::STRING);
      } else {
        output_elem_type->set_elem_type(TensorProto::INT64);
      }
    }));

static constexpr const char* SVMRegressor_ver1_doc = R"DOC(
    Support Vector Machine regression prediction and one-class SVM anomaly detection.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(SVMRegressor, 1, OpSchema().FillUsing(SVMRegressor_v1_FillSpec));

static constexpr const char* TreeEnsembleClassifier_ver5_doc = R"DOC(
    This operator is DEPRECATED. Please use TreeEnsemble with provides similar functionality.
    In order to determine the top class, the ArgMax node can be applied to the output of TreeEnsemble.
    To encode class labels, use a LabelEncoder operator.
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
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleClassifier, 5, OpSchema().FillUsing(TreeEnsembleClassifier_v5_FillSpec));

static constexpr const char* TreeEnsembleRegressor_ver5_doc = R"DOC(
    This operator is DEPRECATED. Please use TreeEnsemble instead which provides the same
    functionality.<br>
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
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(TreeEnsembleRegressor, 5, OpSchema().FillUsing(TreeEnsembleRegressor_v5_FillSpec));

static constexpr const char* TreeEnsemble_ver5_doc = R"DOC(
    Tree Ensemble operator.  Returns the regressed values for each input in a batch.
    Inputs have dimensions `[N, F]` where `N` is the input batch size and `F` is the number of input features.
    Outputs have dimensions `[N, num_targets]` where `N` is the batch size and `num_targets` is the number of targets, which is a configurable attribute.

    The encoding of this attribute is split along interior nodes and the leaves of the trees. Notably, attributes with the prefix `nodes_*` are associated with interior nodes, and attributes with the prefix `leaf_*` are associated with leaves.
    The attributes `nodes_*` must all have the same length and encode a sequence of tuples, as defined by taking all the `nodes_*` fields at a given position.

    All fields prefixed with `leaf_*` represent tree leaves, and similarly define tuples of leaves and must have identical length.

    This operator can be used to implement both the previous `TreeEnsembleRegressor` and `TreeEnsembleClassifier` nodes.
    The `TreeEnsembleRegressor` node maps directly to this node and requires changing how the nodes are represented.
    The `TreeEnsembleClassifier` node can be implemented by adding a `ArgMax` node after this node to determine the top class.
    To encode class labels, a `LabelEncoder` or `GatherND` operator may be used.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsemble,
    5,
    OpSchema().FillUsing(TreeEnsemble_v5_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      checkInputRank(ctx, 0, 2);
      auto nodes_splits = ctx.getAttribute("nodes_splits");
      if (nullptr == nodes_splits) {
        fail_shape_inference("Attribute 'nodes_splits' is required.");
      }
      if (nodes_splits->t().dims_size() != 1) {
        fail_shape_inference("Attribute 'nodes_splits' must be 1D.");
      }
      auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
      // Check that input type is same as split type
      if (input_type != nodes_splits->t().data_type()) {
        fail_shape_inference(
            "Attribute 'nodes_splits' must have same type as input. Input type is ",
            input_type,
            " and attribute type is ",
            nodes_splits->t().data_type());
      }

      // Expected nodes_* length
      auto expected_length = nodes_splits->t().dims(0);
      // Validate all nodes_* attributes that are set have the same length and are 1D.
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_featureids"), expected_length, TensorProto_DataType_INT64, true);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_hitrates"), expected_length, TensorProto_DataType_FLOAT, false);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_modes"), expected_length, TensorProto_DataType_UINT8, true);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_truenodeids"), expected_length, TensorProto_DataType_INT64, true);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_falsenodeids"), expected_length, TensorProto_DataType_INT64, true);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_trueleafs"), expected_length, TensorProto_DataType_INT64, true);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_falseleafs"), expected_length, TensorProto_DataType_INT64, true);
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_missing_value_tracks_true"), expected_length, TensorProto_DataType_INT64, false);

      // The set membership values and the splits must have the same type as the input.
      auto membership_values = ctx.getAttribute("membership_values");
      if (nullptr != membership_values && membership_values->t().data_type() != input_type) {
        fail_shape_inference(
            "Attribute 'membership_values' must have same type as input. Input type is ",
            input_type,
            " and attribute type is ",
            membership_values->t().data_type());
      }
      AssertAttributeProtoTypeAndLength(
          ctx.getAttribute("nodes_splits"), expected_length, static_cast<TensorProto_DataType>(input_type), true);

      // Validate all leaf_* attributes that are set have the same length and are 1D.
      auto leaf_targetids = ctx.getAttribute("leaf_targetids");
      auto leaf_weights = ctx.getAttribute("leaf_weights");
      if (nullptr != leaf_targetids && nullptr != leaf_weights) {
        if (leaf_targetids->ints_size() != leaf_weights->t().dims(0)) {
          fail_shape_inference(
              "Attribute 'leaf_targetids' must have same length as attribute 'leaf_weights'. 'leaf_targetids' "
              "length is ",
              leaf_targetids->ints_size(),
              " and 'leaf_weights' length is ",
              leaf_weights->t().dims(0));
        }
      } else {
        fail_shape_inference("Attributes 'leaf_targetids' and 'leaf_weights' must both be set.");
      }

      // Validate weights have same type as input.
      if (leaf_weights->t().data_type() != input_type) {
        fail_shape_inference(
            "Attribute 'leaf_weights' must have same type as input. Input type is ",
            input_type,
            " and attribute type is ",
            leaf_weights->t().data_type());
      }

      checkInputRank(ctx, 0, 2);

      Dim N, E;
      unifyInputDim(ctx, 0, 0, N);
      if (nullptr != ctx.getAttribute("n_targets")) {
        unifyDim(E, ctx.getAttribute("n_targets")->i());
      }
      updateOutputElemType(ctx, 0, input_type);
      updateOutputShape(ctx, 0, {N, E});
    }));

static constexpr const char* ZipMap_ver1_doc = R"DOC(
    Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    ZipMap,
    1,
    OpSchema().FillUsing(ZipMap_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      std::vector<std::string> classlabels_strings;
      bool result = getRepeatedAttribute(ctx, "classlabels_strings", classlabels_strings);
      auto output_map_type = ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
      auto output_value_tensor_type = output_map_type->mutable_value_type()->mutable_tensor_type();
      output_value_tensor_type->set_elem_type(TensorProto::FLOAT);
      output_value_tensor_type->mutable_shape(); // Initialize to scalar
      if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() != 1 && getInputShape(ctx, 0).dim_size() != 2) {
        fail_shape_inference("ZipMap input shape should be 1D or 2D.")
      }
      if (result && !classlabels_strings.empty()) {
        output_map_type->set_key_type(TensorProto::STRING);
      }
      std::vector<int64_t> classlabels_int64s;
      result = getRepeatedAttribute(ctx, "classlabels_int64s", classlabels_int64s);
      if (result && !classlabels_int64s.empty()) {
        output_map_type->set_key_type(TensorProto::INT64);
      }
    }));

} // namespace ONNX_NAMESPACE
#endif
