/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
static constexpr const char* StringConcat_doc =
    R"DOC(StringConcat concatenates string tensors elementwise (with NumPy-style broadcasting support))DOC";
ONNX_OPERATOR_SET_SCHEMA(
    StringConcat,
    20,
    OpSchema().FillUsing(StringConcat_v20_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

static constexpr const char* RegexFullMatch_doc =
    R"DOC(RegexFullMatch performs a full regex match on each element of the input tensor. If an element fully matches the regex pattern specified as an attribute, the corresponding element in the output is True and it is False otherwise. [RE2](https://github.com/google/re2/wiki/Syntax) regex syntax is used.)DOC";
ONNX_OPERATOR_SET_SCHEMA(
    RegexFullMatch,
    20,
    OpSchema().FillUsing(RegexFullMatch_v20_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

static constexpr const char* StringSplit_doc =
    R"DOC(StringSplit splits a string tensor's elements into substrings based on a delimiter attribute and a maxsplit attribute.

The first output of this operator is a tensor of strings representing the substrings from splitting each input string on the `delimiter` substring. This tensor has one additional rank compared to the input tensor in order to store the substrings for each input element (where the input tensor is not empty). Note that, in order to ensure the same number of elements are present in the final dimension, this tensor will pad empty strings as illustrated in the examples below. Consecutive delimiters are not grouped together and are deemed to delimit empty strings, except if the `delimiter` is unspecified or is the empty string (""). In the case where the `delimiter` is unspecified or the empty string, consecutive whitespace characters are regarded as a single separator and leading or trailing whitespace is removed in the output.

The second output tensor represents the number of substrings generated. `maxsplit` can be used to limit the number of splits performed - after the `maxsplit`th split if the string is not fully split, the trailing suffix of input string after the final split point is also added. For elements where fewer splits are possible than specified in `maxsplit`, it has no effect.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringSplit,
    20,
    OpSchema().FillUsing(StringSplit_v20_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      if (!hasInputShape(ctx, 0)) {
        return;
      }
      const TypeProto* input_type = ctx.getInputType(0);
      if (input_type == nullptr || !input_type->has_tensor_type() ||
          input_type->tensor_type().elem_type() != TensorProto::STRING) {
        return;
      }

      // We produce a string tensor per input element. Therefore we have one additional rank with a runtime
      // dependent number of elements. All except the final dimension of the output shape can be inferred directly
      // from the input.
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      propagateShapeFromInputToOutput(ctx, 0, 0);
      getOutputShape(ctx, 0)->add_dim();

      // The output tensor containing the number of substrings has identical shape to the input but produces int32
      // results.
      ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
      propagateShapeFromInputToOutput(ctx, 0, 1);
    }));

static constexpr const char* StringNormalizer_ver10_doc = R"DOC(
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringNormalizer,
    10,
    OpSchema().FillUsing(StringNormalizer_v10_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      output_elem_type->set_elem_type(TensorProto::STRING);
      if (!hasInputShape(ctx, 0)) {
        return;
      }
      TensorShapeProto output_shape;
      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      auto dim_size = input_shape.dim_size();
      // Last axis dimension is unknown if we have stop-words since we do
      // not know how many stop-words are dropped
      if (dim_size == 1) {
        // Unknown output dimension
        output_shape.add_dim();
      } else if (dim_size == 2) {
        // Copy B-dim
        auto& b_dim = input_shape.dim(0);
        if (!b_dim.has_dim_value() || b_dim.dim_value() != 1) {
          fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
        }
        *output_shape.add_dim() = b_dim;
        output_shape.add_dim();
      } else {
        fail_shape_inference("Input shape must have either [C] or [1,C] dimensions where C > 0");
      }
      updateOutputShape(ctx, 0, output_shape);
    }));
} // namespace ONNX_NAMESPACE
