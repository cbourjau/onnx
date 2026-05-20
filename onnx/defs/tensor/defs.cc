// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/tensor/utils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    25,
    OpSchema()
        .FillUsing(Cast_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    25,
    OpSchema()
        .FillUsing(CastLike_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              auto target_type = ctx.getInputType(1);
              if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
                // we cannot create a correct function body without knowing the target element type
                return false;
              }
              auto target_elt_type = target_type->tensor_type().elem_type();
              FunctionBuilder builder(functionProto);
              builder.Add(MakeString(
                              "output = Cast <to= ",
                              static_cast<int64_t>(target_elt_type),
                              ", saturate: int = @saturate> (input)")
                              .c_str());
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    BitCast,
    26,
    OpSchema().FillUsing(BitCast_v26_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto get_bit_size = [](int element_type) -> int {
        switch (element_type) {
          case TensorProto::FLOAT:
          case TensorProto::INT32:
          case TensorProto::UINT32:
            return 32;
          case TensorProto::DOUBLE:
          case TensorProto::INT64:
          case TensorProto::UINT64:
          case TensorProto::COMPLEX64:
            return 64;
          case TensorProto::COMPLEX128:
            return 128;
          case TensorProto::FLOAT16:
          case TensorProto::BFLOAT16:
          case TensorProto::INT16:
          case TensorProto::UINT16:
            return 16;
          case TensorProto::INT8:
          case TensorProto::UINT8:
          case TensorProto::BOOL:
          case TensorProto::FLOAT8E4M3FN:
          case TensorProto::FLOAT8E4M3FNUZ:
          case TensorProto::FLOAT8E5M2:
          case TensorProto::FLOAT8E5M2FNUZ:
          case TensorProto::FLOAT8E8M0:
            return 8;
          case TensorProto::INT4:
          case TensorProto::UINT4:
          case TensorProto::FLOAT4E2M1:
            return 4;
          case TensorProto::INT2:
          case TensorProto::UINT2:
            return 2;
          default:
            return 0;
        }
      };

      // Validate that input and output types have the same bit-width
      auto input_type = ctx.getInputType(0);
      if (input_type && input_type->has_tensor_type()) {
        auto input_element_type = input_type->tensor_type().elem_type();
        auto* to_attr = ctx.getAttribute("to");
        if (to_attr) {
          auto output_element_type = static_cast<int32_t>(to_attr->i());
          int input_element_bit_size = get_bit_size(input_element_type);
          int output_element_bit_size = get_bit_size(output_element_type);
          if (input_element_bit_size != 0 && output_element_bit_size != 0 &&
              input_element_bit_size != output_element_bit_size) {
            fail_shape_inference(
                "BitCast requires input and output types to have the same bit-width, but input type has ",
                input_element_bit_size,
                " bits and output type has ",
                output_element_bit_size,
                " bits.");
          }
        }
      }

      propagateElemTypeFromAttributeToOutput(ctx, "to", 0);

      // Same bit-width: output shape equals input shape
      if (hasNInputShapes(ctx, 1)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    25,
    OpSchema().FillUsing(Reshape_v25_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      bool found;
      TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
      if (!found) {
        return;
      }

      int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

      // Iterate through targetShape, adding dimensions in the outputShape
      // TensorProto. If the targetShape dimension is -1, we do not set the
      // dimension value in this iteration, but we record the Dimension. If
      // targetShape dimension is 0, we attempt to propagate the dimension
      // value/param. If the value cannot be inferred, we set the flag in
      // the unresolveZeros vector. If targetShape dimension is positive, we
      // set the dimension value in the outputShape. We track the product of
      // the dimensions we are setting outputShape in the outputProduct
      // variable. The outputProduct will potentially be used for inferring
      // a dimension marked -1.
      auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      TensorShapeProto::Dimension* negativeOneDim = nullptr;
      const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
      std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
      int64_t outputProduct = 1;
      bool outputProductValid = true;
      for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
        // Add a new dimension to outputShape
        auto new_dim = outputShape->add_dim();
        if (targetShapeProto.dim(i).has_dim_param()) {
          // There is a tricky edge case here. It is possible that the value of
          // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
          // symbol can be erroneous. This should be a very rare scenario and in such a
          // case an option is to turn off data propagation during shape inference.
          new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
          outputProductValid = false;
        } else {
          if (!targetShapeProto.dim(i).has_dim_value()) {
            outputProductValid = false;
            // treat this dim as unknown dim
            continue;
          }

          const auto dim_value = targetShapeProto.dim(i).dim_value();

          if (dim_value == -1) {
            // Check if multiple -1's. If not, set negativeOneDim, marking
            // this dimension to potentially be filled in later.
            if (negativeOneDim) {
              fail_shape_inference("Target shape may not have multiple -1 dimensions.");
            }
            negativeOneDim = new_dim;
          } else if (dim_value == 0) {
            // Check if data input has a shape and if the index i is within
            // its bounds. If these conditions are satisfied, any dimension
            // value/param should be propagated. If dimension value cannot be
            // inferred, set the corresponding  unresolvedZeros flag to true.
            // If allowzero is set however, do not propagate values, since output
            // dimension is explicitly zero.
            if (allowzero == 0) {
              unresolvedZeros[i] = true;
              if (dataInputTensorType.has_shape()) {
                if (i >= dataInputTensorType.shape().dim_size()) {
                  fail_shape_inference("Invalid position of 0.");
                }
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
                  new_dim->set_dim_value(input_dim_value);
                  if (checked_mul_overflow(outputProduct, input_dim_value, &outputProduct)) {
                    fail_shape_inference("Dimension product overflow in Reshape");
                  }
                  unresolvedZeros[i] = false;
                } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                  new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                }
              }
            } else {
              new_dim->set_dim_value(dim_value);
              if (checked_mul_overflow(outputProduct, dim_value, &outputProduct)) {
                fail_shape_inference("Dimension product overflow in Reshape");
              }
            }
          } else if (dim_value > 0) {
            // Set the dimension value to dim_value
            new_dim->set_dim_value(dim_value);
            if (checked_mul_overflow(outputProduct, dim_value, &outputProduct)) {
              fail_shape_inference("Dimension product overflow in Reshape");
            }
          } else {
            // Check if value is less than -1; fail if so
            fail_shape_inference("Invalid dimension value: ", dim_value);
          }
        }
      }
      // If negativeOneDim has been set, we attempt to infer its value. This
      // can be done if all dimension values for the data input tensor shape
      // are known other than the ones corresponding to unresolvedZeros
      // flags.
      if (negativeOneDim && outputProductValid) {
        // First, attempt to compute product of data input shape dimensions
        // that are not marked by unresolvedZeros. If not possible, set the
        // inputProductValid flag to false.
        if (!outputProduct) {
          fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
        }
        int64_t inputProduct = 1;
        bool inputProductValid = true;
        if (!dataInputTensorType.has_shape()) {
          inputProductValid = false;
        } else {
          for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              if (checked_mul_overflow(inputProduct, dataInputTensorType.shape().dim(i).dim_value(), &inputProduct)) {
                fail_shape_inference("Dimension product overflow in Reshape");
              }
            } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
              inputProductValid = false;
              break;
            }
          }
        }
        if (inputProductValid) {
          if (inputProduct % outputProduct != 0) {
            fail_shape_inference("Dimension could not be inferred: incompatible shapes");
          }
          negativeOneDim->set_dim_value(inputProduct / outputProduct);
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    25,
    OpSchema()
        .FillUsing(Shape_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
          int64_t start = getAttribute(ctx, "start", 0);
          if (start < 0)
            start += rank;
          start = (start < 0) ? 0 : (start > rank) ? rank : start;
          int64_t end = getAttribute(ctx, "end", rank);
          if (end < 0)
            end += rank;
          end = (end < 0) ? 0 : (end > rank) ? rank : end;
          output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            int64_t rank = static_cast<int64_t>(input_shape.dim_size());
            int64_t start = getAttribute(ctx, "start", 0);
            if (start < 0)
              start += rank;
            start = (start < 0) ? 0 : (start > rank) ? rank : start;
            int64_t end = getAttribute(ctx, "end", rank);
            if (end < 0)
              end += rank;
            end = (end < 0) ? 0 : (end > rank) ? rank : end;
            TensorShapeProto output_shape;
            for (int64_t d = start; d < end; ++d) {
              *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
            }
            ctx.addOutputData(0, std::move(output_shape));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    25,
    OpSchema()
        .FillUsing(Size_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          if (input_data != nullptr) {
            TensorShapeProto tsp;
            tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    13,
    OpSchema()
        .FillUsing(Concat_v13_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto numInputs = ctx.getNumInputs();
          if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
            return;
          }

          auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            fail_shape_inference("Required attribute axis is missing");
          }
          int axis = static_cast<int>(axisAttr->i());
          if (axis < -rank || axis >= rank) {
            fail_shape_inference("axis must be in [-rank, rank-1].");
          }
          if (axis < 0) {
            axis += rank;
          }

          if (numInputs == 1) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
            return;
          }

          bool all_lengths_known = true;
          int total_length = 0;

          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int64_t i = 0; i < rank; ++i) {
            output_shape->add_dim();
          }

          for (size_t i = 0; i < numInputs; i++) {
            const auto& shape = ctx.getInputType(i)->tensor_type().shape();
            if (shape.dim_size() != rank) {
              fail_shape_inference(
                  "All inputs to Concat must have same rank. Input ", i, " has rank ", shape.dim_size(), " != ", rank);
            }
            for (int j = 0; j < rank; j++) {
              if (j == axis) {
                if (shape.dim(j).has_dim_value()) {
                  total_length += static_cast<int>(shape.dim(j).dim_value());
                } else {
                  all_lengths_known = false;
                }
              } else {
                auto& output_dim = *output_shape->mutable_dim(j);
                const auto& input_dim = shape.dim(j);
                mergeInDimensionInfo(input_dim, output_dim, j);
              }
            }
          }

          if (all_lengths_known) {
            output_shape->mutable_dim(axis)->set_dim_value(total_length);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (!axisIsZero(ctx)) {
            return;
          }
          TensorShapeProto tsp;
          for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
            const auto input_data = ctx.getInputData(i);
            if (input_data == nullptr) {
              return;
            }
            for (int j = 0; j < input_data->dim_size(); ++j) {
              *tsp.add_dim() = input_data->dim(j);
            }
          }
          if (tsp.dim_size() > 0) {
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

static constexpr const char* Split_ver18_doc =
    R"DOC(Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    18,
    OpSchema().FillUsing(Split_v18_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
        propagateElemTypeFromInputToOutput(ctx, 0, i);
      }
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      const auto& shape = ctx.getInputType(0)->tensor_type().shape();
      int rank = shape.dim_size();
      int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
      if (axis < -rank || axis >= rank) {
        fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
      }
      if (axis < 0) {
        axis += rank;
      }
      const auto& split_dim = shape.dim(axis);
      if (!split_dim.has_dim_value()) {
        for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
          *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
          ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
        }
        return;
      }
      int split_dim_value = static_cast<int>(split_dim.dim_value());

      std::vector<int64_t> split;
      const auto num_outputs_attr = ctx.getAttribute("num_outputs");
      if (ctx.hasInput(1) && num_outputs_attr) {
        fail_shape_inference("Both 'split' input and 'num_outputs' attribute were given");
      }
      if (ctx.hasInput(1)) { //'split' is input
        auto split_proto = ctx.getInputData(1);
        if (split_proto == nullptr) {
          // skip if split is not an initializer
          return;
        }
        split = ParseData<int64_t>(split_proto);
        if (split.size() != ctx.getNumOutputs()) {
          fail_shape_inference(
              "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
        }
        int64_t total_dim = 0;
        for (int64_t d : split) {
          total_dim += d;
        }
        if (total_dim != split_dim_value) {
          fail_shape_inference(
              "Mismatch between the sum of 'split' (",
              total_dim,
              ") and the split dimension of the input (",
              split_dim_value,
              ")");
        }
      } else { // no value available for 'split'
        if (num_outputs_attr) {
          const auto num_outputs = num_outputs_attr->i();
          if (num_outputs < 1) {
            fail_shape_inference("Attribute `num_outputs` value cannot be lower than 1");
          }
          if (split_dim_value % num_outputs == 0) { // tensor is evenly splittable
            int chunk_size = split_dim_value / num_outputs;
            split.resize(num_outputs, chunk_size);
          } else { // tensor needs to be split unevenly
            int chunk_size = (split_dim_value / num_outputs) + 1;
            int last_chunk_size = split_dim_value - (chunk_size * (num_outputs - 1));
            split.resize(num_outputs - 1, chunk_size);
            split.push_back(last_chunk_size);
          }
        } else {
          fail_shape_inference("Neither 'split' input nor 'num_outputs' attribute has been given");
        }
      }
      for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
        *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
        ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
      }
    }));

static constexpr const char* Slice_ver13_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
of its input `data` tensor.

An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
in `[0, ... r-1]` where `r = rank(input)` as follows:

If `axes` are omitted, they are set to `[0, ..., r-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
`dims` are the dimensions of `input` and `steps[i] = 1`.

All negative elements of `axes` are made non-negative by adding `r` to them, where
`r =rank(input)`.

All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
`starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
and `[0, dims[axes[i]]-1]` for negative stepping.

The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
`ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
is clamped to `[-1, dims[axes[i]]-1]`.

Finally, `steps[axes[i]] = steps[i]`.

For slicing to the end of a dimension with unknown size, it is recommended to pass
in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

Example 1:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
axes = [0, 1]
starts = [1, 0]
ends = [2, 3]
steps = [1, 2]
result = [
    [5, 7],
]
```

Example 2:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
starts = [0, 1]
ends = [-1, 1000]
result = [
    [2, 3, 4],
]
```
)DOC";

static void processSliceInputs(const int64_t input_dim_size_or_value, int64_t& start, int64_t& end, int64_t step) {
  // process step
  if (step == 0) {
    fail_shape_inference("'step' cannot be 0 for Slice");
  }
  // Empty dimension: clamp bounds are invalid when dimension size is 0,
  // so short-circuit to produce a zero-length output.
  if (input_dim_size_or_value == 0) {
    start = 0;
    end = 0;
    return;
  }
  // process start
  if (start < 0)
    start += input_dim_size_or_value;
  if (step < 0)
    start = std::clamp(start, static_cast<int64_t>(0), input_dim_size_or_value - 1);
  else
    start = std::clamp(start, static_cast<int64_t>(0), input_dim_size_or_value);
  // process end
  if (end < 0)
    end += input_dim_size_or_value;
  if (step < 0)
    end = std::clamp(end, static_cast<int64_t>(-1), input_dim_size_or_value - 1);
  else
    end = std::clamp(end, static_cast<int64_t>(0), input_dim_size_or_value);
}

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    13,
    OpSchema()
        .FillUsing(Slice_v13_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          size_t num_inputs = ctx.getNumInputs();
          if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
            fail_type_inference("Slice op must have either three, four or five inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if
          //     1. 2nd and 3rd input data (starts, ends) are available.
          // and 2. 4th and 5th optional input (axes, steps) are either not set,
          // or set and is initializer.
          const TensorProto* startsInitializer = ctx.getInputData(1);
          const TensorProto* endsInitializer = ctx.getInputData(2);
          const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
          const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

          if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
              (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
            const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
            // we can infer the output rank - it never changes
            for (int i = 0; i < input_rank; ++i) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            }
            return;
          }

          // don't know data_type- can't proceed
          if (!startsInitializer->has_data_type())
            return;

          auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
            std::vector<int64_t> vec;
            if (initializer->data_type() == TensorProto::INT64) {
              const auto data = ParseData<int64_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else if (initializer->data_type() == TensorProto::INT32) {
              const auto data = ParseData<int32_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else {
              // unaccepted data type
              fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
            }
            return vec;
          };

          std::vector<int64_t> starts = get_initializer_data(startsInitializer);
          std::vector<int64_t> ends = get_initializer_data(endsInitializer);

          if (starts.size() != ends.size()) {
            fail_shape_inference("Incorrect or missing input value for starts and ends");
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();
          std::vector<int64_t> axes(starts.size());
          if (!axesInitializer) {
            std::iota(axes.begin(), axes.end(), 0);
          } else {
            axes = get_initializer_data(axesInitializer);
            if (axes.size() != starts.size()) {
              fail_shape_inference("Input axes has incorrect length");
            }
          }
          checkAxesRange(axes, input_rank);
          adjustNegativeAxes(axes, input_rank);
          checkDuplicateAxes(axes, input_rank);
          std::vector<int64_t> steps;
          if (!stepsInitializer) {
            steps = std::vector<int64_t>(starts.size(), 1);
          } else {
            steps = get_initializer_data(stepsInitializer);
            if (steps.size() != axes.size()) {
              fail_shape_inference("Input steps has incorrect length");
            }
          }

          for (int i = 0; i < input_rank; ++i) {
            // first update rank of output dim
            auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            const auto& input_dim = input_shape.dim(i);
            if (input_dim.has_dim_value()) {
              output_dim->set_dim_value(input_dim.dim_value());
            } else if (input_dim.has_dim_param()) {
              output_dim->set_dim_param(input_dim.dim_param());
            }
          }

          size_t axes_size = axes.size();
          for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
            auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

            auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(static_cast<int>(axis));

            // input dim value is missing - cannot perform shape inference for
            // this axis
            if (!input_dim.has_dim_value()) {
              // Clear any previously propagated dim_param and leave this dimension "empty",
              // before moving on to the next dimension
              ctx.getOutputType(0)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->mutable_dim(static_cast<int>(axis))
                  ->clear_dim_param();
              continue;
            }
            auto start = starts[axis_index];
            auto end = ends[axis_index];
            auto step = steps[axis_index];
            processSliceInputs(input_dim.dim_value(), start, end, step);

            // find output dim value for this axis
            auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
            if (temp < 0)
              temp = 0;

            // assign output value
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->mutable_dim(static_cast<int>(axis))
                ->set_dim_value(temp);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          const auto starts = ctx.getInputData(1);
          const auto ends = ctx.getInputData(2);
          bool axes_specified = ctx.getNumInputs() >= 4;
          bool steps_specified = ctx.getNumInputs() >= 5;

          const TensorShapeProto* axes = nullptr;
          const TensorShapeProto* steps = nullptr;
          if (axes_specified) {
            axes = ctx.getInputData(3);
            if (axes == nullptr) {
              return;
            }
          }
          if (steps_specified) {
            steps = ctx.getInputData(4);
            if (steps == nullptr) {
              return;
            }
          }

          if (input_data == nullptr || starts == nullptr || ends == nullptr) {
            return;
          }
          if (starts->dim_size() != ends->dim_size()) {
            fail_shape_inference(
                "Input rank for starts and ends should be the same: (",
                starts->dim_size(),
                ") vs (",
                ends->dim_size(),
                ").");
          }
          // Only supports axis = 0 since the data comes from Shape
          if ((!axes_specified || (axes->dim_size() == 1 && axes->dim(0).dim_value() == 0)) &&
              starts->dim_size() == 1 && ends->dim_size() == 1) {
            auto start = starts->dim(0).dim_value();
            auto end = ends->dim(0).dim_value();
            int64_t step = 1; // Default step is 1
            if (steps_specified) {
              if (steps->dim_size() != 1) {
                return;
              }
              if (!steps->dim(0).has_dim_value()) {
                return;
              }
              step = steps->dim(0).dim_value();
            }
            processSliceInputs(input_data->dim_size(), start, end, step);

            TensorShapeProto tsp;
            if (step > 0) {
              for (int i = start; i < end; i += step) {
                *tsp.add_dim() = input_data->dim(i);
              }
            } else {
              for (int i = start; i > end; i += step) {
                *tsp.add_dim() = input_data->dim(i);
              }
            }
            if (tsp.dim_size() > 0) {
              ctx.addOutputData(0, std::move(tsp));
            }
          }
        }));

const char* Transpose_doc = R"DOC(
Returns a transpose of the input tensor. (Similar to `numpy.transpose`).
The optional attribute `perm` must be a permutation of the dimensions of
the input tensor. Axis `i` of the output tensor corresponds to the axis
`perm[i]` of the input tensor.
For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3),
the output shape will be (2, 1, 3).
When perm=(1, 2, 0), given an input tensor of shape (1, 2, 3),
the output shape will be (2, 3, 1).
If the attribute `perm` is omitted, its default value is `(n-1, ..., 0)`,
where `n` is the rank of the input tensor.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    25,
    OpSchema().FillUsing(Transpose_v25_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }
      auto input_type = ctx.getInputType(0);
      const TensorShapeProto& shape = input_type->tensor_type().shape();
      std::vector<int64_t> perm;
      bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
      if (!has_perm_attr) {
        perm.reserve(shape.dim_size());
        for (int i = shape.dim_size() - 1; i >= 0; --i)
          perm.push_back(i);
      } else if (!perm.empty()) {
        // check if every index is valid
        std::vector<bool> seen(shape.dim_size(), false);
        for (int64_t fromDimIndex : perm) {
          if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
            std::ostringstream oss;
            oss << "Invalid attribute perm {" << perm[0];
            for (size_t i = 1; i != perm.size(); ++i) {
              oss << ", " << perm[i];
            }
            oss << "}, input shape = {";
            if (shape.dim_size() > 0) {
              oss << shape.dim(0).dim_value();
              for (int i = 1; i != shape.dim_size(); ++i) {
                oss << ", " << shape.dim(i).dim_value();
              }
              oss << "}";
            }
            fail_type_inference(oss.str());
          } else {
            // check if any perm is repeated
            if (seen[fromDimIndex]) {
              fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
            }
            seen[fromDimIndex] = true;
          }
        }
      }

      getOutputShape(ctx, 0);

      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      for (int64_t i : perm) {
        appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
      }
    }));

static constexpr const char* Scatter_ver11_doc = R"DOC(
This operator is deprecated. Please use ScatterElements, which provides the same functionality.

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    11,
    OpSchema().FillUsing(Scatter_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 1)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

static constexpr const char* ScatterND_ver18_doc = R"DOC(
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
`indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[tuple(indices[idx])] = updates[idx]
```

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. This ensures that the output value does not depend on the iteration order.
When `reduction` is set to some reduction function `f`, `output` is calculated as follows:

```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[tuple(indices[idx])] = f(output[tuple(indices[idx])], updates[idx])
```

where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherND.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
data    = [1, 2, 3, 4, 5, 6, 7, 8]
indices = [[4], [3], [1], [7]]
updates = [9, 10, 11, 12]
output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
indices = [[0], [2]]
updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    18,
    OpSchema().FillUsing(ScatterND_v18_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 1)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

static constexpr const char* ScatterElements_ver18_doc = R"DOC(
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case, the update
corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = updates[i][j] if axis = 0,
output[i][indices[i][j]] = updates[i][j] if axis = 1,
```
When `reduction` is set to some reduction function `f`, the update corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,
```
where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
data = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
indices = [
    [1, 0, 2],
    [0, 2, 1],
]
updates = [
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2],
]
output = [
    [2.0, 1.1, 0.0]
    [1.0, 0.0, 2.2]
    [0.0, 2.1, 1.2]
]
```
Example 2:
```
data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
indices = [[1, 3]]
updates = [[1.1, 2.1]]
axis = 1
output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    18,
    OpSchema().FillUsing(ScatterElements_v18_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 1)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

static constexpr const char* Gather_ver13_doc = R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

It is an indexing operation that indexes into the input `data` along a single (specified) axis.
Each entry in `indices` produces a `r-1` dimensional slice of the input tensor.
The entire operation produces, conceptually, a `q`-dimensional tensor of `r-1` dimensional slices,
which is arranged into a `q + (r-1)`-dimensional tensor, with the `q` dimensions taking the
place of the original `axis` that is being indexed into.

The following few examples illustrate how `Gather` works for specific shapes of `data`,
`indices`, and given value of `axis`:
| data shape | indices shape | axis | output shape | output equation |
| --- | --- | --- | --- | --- |
| (P, Q) | ( )  (a scalar)   | 0 | (Q)       | output[q] = data[indices, q] |
| (P, Q, R) | ( )  (a scalar)   | 1 | (P, R)       | output[p, r] = data[p, indices, r] |
| (P, Q) | (R, S) | 0 | (R, S, Q) | output[r, s, q] = data[ [indices[r, s], q] |
| (P, Q) | (R, S) | 1 | (P, R, S) | output[p, r, s] = data[ p, indices[r, s]] |

More generally, if `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    13,
    OpSchema()
        .FillUsing(Gather_v13_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }
          const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
          int r = data_shape.dim_size();
          if (r < 1) {
            fail_shape_inference("data tensor must have rank >= 1");
          }
          int q = indices_shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -r || axis >= r) {
            fail_shape_inference("axis must be in [-r, r-1]");
          }
          if (axis < 0) {
            axis += r;
          }
          int out_rank = q + r - 1;
          if (out_rank == 0) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          }
          for (int i = 0; i < out_rank; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                                  : // i < axis < r
                (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                            : // i - axis < q
                data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { GatherOp13DataPropagator(ctx); }));

static constexpr const char* GatherElements_ver13_doc = R"DOC(

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```
Example 2:
```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GatherElements,
    13,
    OpSchema().FillUsing(GatherElements_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // propagate indices' shape to output if it exists
      if (hasInputShape(ctx, 1)) {
        propagateShapeFromInputToOutput(ctx, 1, 0);
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    25,
    OpSchema()
        .FillUsing(Squeeze_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          size_t num_inputs = ctx.getNumInputs();
          bool axes_not_specified = false;

          if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
            auto axes_proto = ctx.getInputData(1);
            if (axes_proto == nullptr) {
              // skip if axes is not an initializer
              return;
            }
            axes = ParseData<int64_t>(axes_proto);
          } else {
            // axes not specified
            axes_not_specified = true;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          checkAxesRange(axes, input_ndim);
          adjustNegativeAxes(axes, input_ndim);

          for (int i = 0; i < input_ndim; ++i) {
            if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
              // if dim has a symbolic value and the axes spec want to act on all dims,
              // return early because we can't infer the shape
              return;
            }
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int i = 0; i < input_ndim; ++i) {
            if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
              // if axes not specified, do not keep shape if the dimension is equal to one
              continue;
            } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
              // if axes wants to explicitly act on this dim, fail explicitly only if the
              // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
              // the dim is symbolic, runtime engines should check that the dimension is
              // actually 1 when the op is evaluated
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    25,
    OpSchema()
        .FillUsing(Unsqueeze_v25_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> axes;
          auto axes_proto = ctx.getInputData(1);
          if (axes_proto == nullptr) {
            // skip if axes is not an initializer
            return;
          }
          axes = ParseData<int64_t>(axes_proto);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          checkAxesRange(axes, output_ndim);
          adjustNegativeAxes(axes, output_ndim);
          checkDuplicateAxes(axes, output_ndim);
          // sort after correcting negative axes values (if any)
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    13,
    OpSchema().FillUsing(SpaceToDepth_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto blocksize = getAttribute(ctx, "blocksize", 0);
      if (blocksize <= 0) {
        fail_shape_inference("Blocksize must be positive");
      }
      if (hasInputShape(ctx, 0)) {
        auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() == 4) {
          updateOutputShape(
              ctx,
              0,
              {input_shape.dim(0),
               input_shape.dim(1) * (blocksize * blocksize),
               input_shape.dim(2) / blocksize,
               input_shape.dim(3) / blocksize});
        } else {
          fail_shape_inference("Input tensor must be 4-dimensional");
        }
      }
    }));

static constexpr const char* DepthToSpace_ver13_doc =
    R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
```

In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    13,
    OpSchema().FillUsing(DepthToSpace_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto blocksize = getAttribute(ctx, "blocksize", 0);
      if (blocksize <= 0) {
        fail_shape_inference("Blocksize must be positive");
      }
      if (hasInputShape(ctx, 0)) {
        auto& input_shape = getInputShape(ctx, 0);
        if (input_shape.dim_size() == 4) {
          updateOutputShape(
              ctx,
              0,
              {input_shape.dim(0),
               input_shape.dim(1) / (blocksize * blocksize),
               input_shape.dim(2) * blocksize,
               input_shape.dim(3) * blocksize});
        } else {
          fail_shape_inference("Input tensor must be 4-dimensional");
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    13,
    OpSchema().FillUsing(Tile_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference

      // Needs at least the first input to proceed
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      const auto input_rank = input_shape.dim_size();

      const auto repeats_inputs = ctx.getInputData(1);

      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

      if (nullptr != repeats_inputs && hasNInputShapes(ctx, 2)) {
        // shape inference is possible only when 'repeats' is an initializer
        const auto& repeats_shape = ctx.getInputType(1)->tensor_type().shape();
        if (repeats_shape.dim_size() != 1 || repeats_inputs->data_type() != TensorProto::INT64) {
          fail_shape_inference("'Repeats' input must be 1D tensor of type int64");
        }

        const auto repeats_data = ParseData<int64_t>(repeats_inputs);

        if (repeats_data.size() != static_cast<size_t>(input_rank)) {
          fail_shape_inference(
              "'Repeats' input has incorrect number of values. "
              "The number of values in 'repeats' must be equal "
              "to the number of input dimensions.");
        }

        for (int i = 0; i < input_rank; ++i) {
          const auto& input_dim = input_shape.dim(i);
          auto output_dim = output_shape->add_dim();
          if (input_dim.has_dim_value()) {
            output_dim->set_dim_value(input_dim.dim_value() * repeats_data[i]);
          }
        }
      } else {
        // Infer output shape's rank in any case (if repeats data is not
        // available)
        auto output_shape_0 = getOutputShape(ctx, 0);
        for (int i = 0; i < input_rank; ++i) {
          output_shape_0->add_dim();
        }
      }
      return;
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    10,
    OpSchema().FillUsing(Upsample_v10_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      resizeShapeInference_opset7_to_10(ctx);
    }));

static constexpr const char* Resize_ver19_doc = R"DOC(
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
```
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
```
if input \"sizes\" is not specified.
)DOC";

static constexpr const char* Resize_ver19_attr_coordinate_transformation_mode_doc = R"DOC(
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote `x_resized` as the coordinate of axis x in the resized tensor,
 `x_original` as the coordinate of axis x in the original tensor,
 `length_original` as the length of the original tensor in axis x,
 `length_resized` as the length of the resized tensor in axis x,
 `scale = length_resized / length_original`,
 `output_width` the target length on the axis x which can be a fractional number when it is calculated out of a scale factor,
 and `output_width_int` the effective output width as an integer.

if coordinate_transformation_mode is `"half_pixel"`,
```
x_original = (x_resized + 0.5) / scale - 0.5
```

if coordinate_transformation_mode is `"half_pixel_symmetric"`,
```
adjustment = output_width_int / output_width
center = input_width / 2
offset = center * (1 - adjustment)
x_ori = offset + (x + 0.5) / scale - 0.5
```

if coordinate_transformation_mode is `"pytorch_half_pixel"`,
```
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0
```

if coordinate_transformation_mode is `"align_corners"`,
```
x_original = x_resized * (length_original - 1) / (length_resized - 1)
```

if coordinate_transformation_mode is `"asymmetric"`,
```
x_original = x_resized / scale
```

if coordinate_transformation_mode is `"tf_crop_and_resize"`,
```
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)
```
.)DOC";

static constexpr const char* Resize_ver19_attr_keep_aspect_ratio_policy_doc = R"DOC(
This attribute describes how to interpret the `sizes` input with regard to keeping the original aspect ratio of the input, and it is not applicable when
the `scales` input is used.

Given a set of `sizes`, associated with a subset of `axes` (explicitly provided or default), and assuming `d = axes[i]`, with `i` being the index of the provided `sizes`.

If `keep_aspect_ratio_policy` is `"stretch"`, the original aspect ratio is disregarded, and the input is resized to the specified size:
`out_size[d] = sizes[i]`

If `keep_aspect_ratio_policy` is `"not_larger"`, the sizes are adjusted so that no extent of the output is larger than the specified size, while keeping the original aspect ratio:
```
scale = Min(sizes[i] / in_size[d])
out_size[d] = round_int(scale * in_size[d])
```

If `keep_aspect_ratio_policy` is `"not_smaller"`, the sizes are adjusted so that no extent of the output is smaller than the specified size, while keeping the original aspect ratio:
```
scale = Max(sizes[i] / in_size[d])
out_size[d] = round_int(scale * in_size[d])
```

For non-resizable axes (those not specified in `axes`), the output size will be equal to the input size.

Note: `round_int` stands for computing the nearest integer value, rounding halfway cases up.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    19,
    OpSchema().FillUsing(Resize_v19_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      resizeShapeInference_opset18_to_19(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    22,
    OpSchema().FillUsing(GridSample_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      gridSampleShapeInference(ctx);
    }));

static constexpr const char* AffineGrid_ver20_doc = R"DOC(
Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
(https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
```
[r00, r01, r02, t0]   [x]   [x']
[r10, r11, r12, t1] * [y] = [y']
[r20, r21, r22, t2]   [z]   [z']
[0,   0,   0,   1 ]   [1]   [1 ]
```
where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.

Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
The output `grid` contains positions in the output space.

When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
```
v            v            v            v
|-------------------|------------------|
-1                  0                  1
```
When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
```
    v        v         v         v
|------------------|-------------------|
-1                 0                   1
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    AffineGrid,
    20,
    OpSchema()
        .FillUsing(AffineGrid_v20_FillSpec)
        .FunctionBody(R"ONNX(
        {
          # naming one: 1, one_f: 1.0, one_1d: [1], one_f_1d: [1.0]
          one = Constant <value_int: int=1> ()
          two = Constant <value_int: int=2> ()
          zero = Constant <value_int: int=0> ()
          four = Constant <value_int: int=4> ()
          one_1d = Constant <value_ints: ints = [1]> ()
          zero_1d = Constant <value_ints: ints = [0]> ()

          minus_one = Constant <value_int: int=-1> ()
          minus_one_f = CastLike (minus_one, theta)
          zero_f = CastLike (zero, theta)
          one_f = CastLike (one, theta)
          two_f = CastLike (two, theta)

          constant_align_corners = Constant <value_int: int=@align_corners> ()
          constant_align_corners_equal_zero = Equal (constant_align_corners, zero)

          size_ndim = Size (size)
          condition_is_2d = Equal (size_ndim, four)

          N, C, D, H, W = If (condition_is_2d) <
              then_branch = g1 () => (N_then, C_then, D_then, H_then, W_then) {
                  N_then, C_then, H_then, W_then = Split <num_outputs: int=4> (size)
                  D_then = Identity (one_1d)
              },
              else_branch = g2 () => (N_else, C_else, D_else, H_else, W_else) {
                  N_else, C_else, D_else, H_else, W_else = Split <num_outputs: int=5> (size)
              }
          >
          size_NCDHW = Concat <axis=0> (N, C, D, H, W)

          theta_3d = If (condition_is_2d) <
              then_branch = g3 () => (theta_then) { # theta: N by 2 by 3 => N by 3 by 4
                  # use of thetaN23 is a way to make shape inference happy when theta is N by 3 by 4.
                  gather_idx_6 = Constant <value_ints: ints = [0, 1, 2, 0, 1, 2]> ()
                  shape_23 = Constant <value_ints: ints = [2, 3]> ()
                  gather_idx_23 = Reshape (gather_idx_6, shape_23)
                  shape_N23 = Concat <axis=0>(N, shape_23)
                  gather_idx_N23 = Expand (gather_idx_23, shape_N23)
                  thetaN23 = GatherElements <axis=2> (theta, gather_idx_N23) # N by 2 by 3 => N by 3 by 2

                  r1, r2 = Split <axis: int=1, num_outputs: int=2> (thetaN23) # N by 1 by 3
                  r1_ = Squeeze (r1) # N by 3
                  r2_ = Squeeze (r2)
                  r11, r12, t1 = Split <axis: int=1, num_outputs: int=3> (r1_) # N by 1
                  r21, r22, t2 = Split <axis: int=1, num_outputs: int=3> (r2_)

                  r11_shape = Shape (r21)
                  float_zero_1d_ = ConstantOfShape (r11_shape) # N by 1
                  float_zero_1d = CastLike (float_zero_1d_, theta)
                  float_one_1d = Add (float_zero_1d, one_f) # N by 1

                  R1 = Concat <axis=1>(r11, r12, float_zero_1d, t1) # N by 4
                  R2 = Concat <axis=1>(r21, r22, float_zero_1d, t2)
                  R3 = Concat <axis=1>(float_zero_1d, float_zero_1d, float_one_1d, float_zero_1d)

                  R1_ = Unsqueeze (R1, one_1d) # N by 1 by 4
                  R2_ = Unsqueeze (R2, one_1d)
                  R3_ = Unsqueeze (R3, one_1d)
                  theta_then = Concat <axis=1> (R1_, R2_, R3_) # N by 3 by 4
                  # theta_then = Identity (theta)
              },
              else_branch = g4 () => (theta_else) {
                  theta_else = Identity (theta)
              }
          >

          two_1d = Constant <value_ints=[2]> ()
          three_1d = Constant <value_ints=[3]> ()
          five_1d = Constant <value_ints=[5]> ()
          constant_D_H_W_shape = Slice (size_NCDHW, two_1d, five_1d) # [N, C, D, H, W] => [D, H, W]
          zeros_D_H_W_ = ConstantOfShape (constant_D_H_W_shape)
          zeros_D_H_W = CastLike (zeros_D_H_W_, theta)
          ones_D_H_W = Add (zeros_D_H_W, one_f)

          D_float = CastLike (D, zero_f)
          H_float = CastLike (H, zero_f)
          W_float = CastLike (W, zero_f)
          start_d, step_d, start_h, step_h, start_w, step_w = If (constant_align_corners_equal_zero) <
              then_branch = h1 () => (start_d_then, step_d_then, start_h_then, step_h_then, start_w_then, step_w_then) { # => (float, float, float, float, float, float)
                  step_d_then = Div (two_f, D_float)
                  step_h_then = Div (two_f, H_float)
                  step_w_then = Div (two_f, W_float)

                  step_d_half = Div (step_d_then, two_f)
                  start_d_then = Add (minus_one_f, step_d_half)

                  step_h_half = Div (step_h_then, two_f)
                  start_h_then = Add (minus_one_f, step_h_half)

                  step_w_half = Div (step_w_then, two_f)
                  start_w_then = Add (minus_one_f, step_w_half)
              },
              else_branch = h2 () => (start_d_else, step_d_else, start_h_else, step_h_else, start_w_else, step_w_else) { # => (float, float, float, float, float, float)
                  D_float_nimus_one = Sub (D_float, one_f)
                  H_float_nimus_one = Sub (H_float, one_f)
                  W_float_nimus_one = Sub (W_float, one_f)
                  # avoid divide by 0
                  D_equals_one = Equal (D, one)
                  step_d_else = If (D_equals_one) <
                      then_branch = g5 () => (step_d_else_then) {
                          step_d_else_then = Identity (zero_f)
                      },
                      else_branch = g6 () => (step_d_else_else) {
                          step_d_else_else = Div (two_f, D_float_nimus_one)
                      }
                  >
                  step_h_else = Div (two_f, H_float_nimus_one)
                  step_w_else = Div (two_f, W_float_nimus_one)
                  start_d_else = Identity (minus_one_f)
                  start_h_else = Identity (minus_one_f)
                  start_w_else = Identity (minus_one_f)
              }
          >
          grid_w_steps_int = Range (zero, W, one)
          grid_w_steps_float = CastLike (grid_w_steps_int, step_w)
          grid_w_steps = Mul (grid_w_steps_float, step_w)
          grid_w_0 = Add (start_w, grid_w_steps)

          grid_h_steps_int = Range (zero, H, one)
          grid_h_steps_float = CastLike (grid_h_steps_int, step_h)
          grid_h_steps = Mul (grid_h_steps_float, step_h)
          grid_h_0 = Add (start_h, grid_h_steps)

          grid_d_steps_int = Range (zero, D, one)
          grid_d_steps_float = CastLike (grid_d_steps_int, step_d)
          grid_d_steps = Mul (grid_d_steps_float, step_d)
          grid_d_0 = Add (start_d, grid_d_steps)

          zeros_H_W_D = Transpose <perm = [1, 2, 0]> (zeros_D_H_W)
          grid_d_1 = Add (zeros_H_W_D, grid_d_0)
          grid_d = Transpose <perm = [2, 0, 1]> (grid_d_1)

          zeros_D_W_H = Transpose <perm = [0, 2, 1]> (zeros_D_H_W)
          grid_h_1 = Add (zeros_D_W_H, grid_h_0)
          grid_h = Transpose <perm = [0, 2, 1]> (grid_h_1)

          grid_w = Add (grid_w_0, zeros_D_H_W)

          grid_w_usqzed = Unsqueeze (grid_w, minus_one)
          grid_h_usqzed = Unsqueeze (grid_h, minus_one)
          grid_d_usqzed = Unsqueeze (grid_d, minus_one)
          ones_D_H_W_usqzed = Unsqueeze (ones_D_H_W, minus_one)
          original_grid = Concat <axis=-1> (grid_w_usqzed, grid_h_usqzed, grid_d_usqzed, ones_D_H_W_usqzed)

          constant_shape_DHW_4 = Constant <value_ints: ints = [-1, 4]> ()
          original_grid_DHW_4 = Reshape (original_grid, constant_shape_DHW_4)
          original_grid_4_DHW_ = Transpose (original_grid_DHW_4)

          original_grid_4_DHW = CastLike (original_grid_4_DHW_, theta_3d)
          grid_N_3_DHW = MatMul (theta_3d, original_grid_4_DHW)
          grid_N_DHW_3 = Transpose <perm = [0, 2, 1]> (grid_N_3_DHW)
          N_D_H_W_3 = Concat <axis=-1> (N, D, H, W, three_1d)
          grid_3d_else_ = Reshape (grid_N_DHW_3, N_D_H_W_3)
          grid_3d = CastLike (grid_3d_else_, theta_3d)

          # grid = Identity (grid_3d)
          grid = If (condition_is_2d) <
              then_branch = g1 () => (grid_then) { # [N, D=1, H, W, 3] => [N, H, W, 2]
                  grid_squeezed = Squeeze (grid_3d, one_1d)  # [N, H, W, 3]
                  grid_then = Slice (grid_squeezed, zero_1d, two_1d, three_1d) # [N, H, W, 2]
              },
              else_branch = g2 () => (grid_else) {
                  grid_else = Identity (grid_3d)
              }
          >
        }
        )ONNX")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          checkInputRank(ctx, 1, 1);

          bool found;
          TensorShapeProto size_proto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          const auto size_length = size_proto.dim_size();
          if (size_length != 4 && size_length != 5) {
            fail_shape_inference("Length of input 'size' is ", size_length, ". It must be 4 for 2D or 5 for 5D.");
          }

          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& N = size_proto.dim(0);
          *output_shape->add_dim() = N;
          // const auto& C = size_proto.dim(1); // C is not used
          if (size_length == 4) {
            // 2D case: size shape (N, C, H, W), output shape (N, C, H, W, 2)
            const auto& H = size_proto.dim(2);
            const auto& W = size_proto.dim(3);
            *output_shape->add_dim() = H;
            *output_shape->add_dim() = W;
            output_shape->add_dim()->set_dim_value(2);
          } else if (size_length == 5) {
            // 3D case: size shape (N, C, D, H, W), output shape (N, C, D, H, W, 3)
            const auto& D = size_proto.dim(2);
            const auto& H = size_proto.dim(3);
            const auto& W = size_proto.dim(4);
            *output_shape->add_dim() = D;
            *output_shape->add_dim() = H;
            *output_shape->add_dim() = W;
            output_shape->add_dim()->set_dim_value(3);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    25,
    OpSchema().FillUsing(Identity_v25_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    11,
    OpSchema().FillUsing(Compress_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto axisAttr = ctx.getAttribute("axis");
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
        int r = indices_shape.dim_size();
        if (r < 1) {
          fail_shape_inference("Indices tensor must have rank >= 1");
        }
        if (axisAttr) {
          int axis = static_cast<int>(axisAttr->i());
          if (axis < -r || axis >= r) {
            fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
          }
          if (axis < 0) {
            axis += r;
          }
          TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          for (int i = 0; i < indices_shape.dim_size(); i++) {
            auto dim = shape->add_dim();
            if (i != axis) {
              *dim = indices_shape.dim(i);
            }
          }
        }
      }
      if (!axisAttr) {
        updateOutputShape(ctx, 0, {Dim()});
      }
    }));

static constexpr const char* OneHot_ver11_doc = R"DOC(
    Produces a one-hot tensor based on inputs.
    The locations represented by the index values in the 'indices' input tensor will have 'on_value'
    and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
    are specified as part of required input argument 'values', which is a two-element tensor of format
    [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
    input tensor. The additional dimension is for one-hot representation. The additional dimension will
    be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
    dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
    dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
    as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
    the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
    output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    11,
    OpSchema().FillUsing(OneHot_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Check that the node has three inputs.
      if (ctx.getNumInputs() != 3) {
        fail_type_inference("OneHot node must have three inputs.");
      }
      // Input 'depth' must be a scalar or a single-element vector.
      // TODO(ONNX): Ideally to match spec for this input only Scalar should
      // be allowed. Making this change now can affect backward
      // compatibility for this op. Since this does not seem like a good
      // justification to update version for this op, allowing both scalar
      // and 1 element vector for now. In future when version update for
      // this op is done we should only allow scalar or change the spec to
      // allow both.
      std::optional<int64_t> depth_value;
      if (hasInputShape(ctx, 1)) {
        auto& depth_shape = getInputShape(ctx, 1);
        if (const TensorProto* depth_data = ctx.getInputData(1)) {
          if (depth_data->data_type() == TensorProto::INT64) {
            depth_value = ParseData<int64_t>(depth_data)[0];
          } else if (depth_data->data_type() == TensorProto::INT32) {
            depth_value = ParseData<int32_t>(depth_data)[0];
          } else if (depth_data->data_type() == TensorProto::FLOAT) {
            depth_value = static_cast<int64_t>(ParseData<float>(depth_data)[0]);
          }
        }
        if (depth_shape.dim_size() != 0 && depth_shape.dim_size() != 1) {
          fail_type_inference("Input 'depth' must be a scalar or rank 1 tensor.");
        }
        if (depth_shape.dim_size() == 1 && depth_shape.dim(0).has_dim_value() && depth_shape.dim(0).dim_value() != 1) {
          fail_type_inference("Input 'depth' must have exactly one element.");
        }
      }
      // Input 'values' must be a two-element vector.
      if (hasInputShape(ctx, 2)) {
        auto& values_shape = getInputShape(ctx, 2);
        if (values_shape.dim_size() != 1) {
          fail_type_inference("Input 'values' must be rank 1 tensor.");
        }
        if (values_shape.dim(0).has_dim_value() && values_shape.dim(0).dim_value() != 2) {
          fail_type_inference("Input 'values' must have exactly two elements.");
        }
      }
      // Set output type to be the same as the third input, 'values'.
      propagateElemTypeFromInputToOutput(ctx, 2, 0);
      // Set the output shape, if input 0 (indices) shape is available.
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
        int r = indices_shape.dim_size();
        if (r < 1) {
          fail_shape_inference("Indices tensor must have rank >= 1");
        }
        int out_rank = r + 1;
        int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
        if (axis < -out_rank || axis >= out_rank) {
          fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
        }
        if (axis < 0) {
          axis += out_rank;
        }
        auto output_shape = getOutputShape(ctx, 0);
        for (int i = 0; i < out_rank; ++i) {
          auto dim = output_shape->add_dim();
          if (i < axis) {
            if (indices_shape.dim(i).has_dim_value()) {
              dim->set_dim_value(indices_shape.dim(i).dim_value());
            } else if (indices_shape.dim(i).has_dim_param()) {
              dim->set_dim_param(indices_shape.dim(i).dim_param());
            }
          } else if (i > axis) {
            if (indices_shape.dim(i - 1).has_dim_value()) {
              dim->set_dim_value(indices_shape.dim(i - 1).dim_value());
            } else if (indices_shape.dim(i - 1).has_dim_param()) {
              dim->set_dim_param(indices_shape.dim(i - 1).dim_param());
            }
          } else if (depth_value) {
            dim->set_dim_value(*depth_value);
          }
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    20,
    OpSchema().FillUsing(IsNaN_v20_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    20,
    OpSchema().FillUsing(IsInf_v20_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      updateOutputElemType(ctx, 0, TensorProto::BOOL);
      if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    16,
    OpSchema().FillUsing(Where_v16_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 1, 0);
      if (hasNInputShapes(ctx, 3)) {
        std::vector<const TensorShapeProto*> shapes;
        shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
        shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
        shapes.push_back(&ctx.getInputType(2)->tensor_type().shape());
        multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    NonZero,
    13,
    OpSchema().FillUsing(NonZero_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      updateOutputElemType(ctx, 0, TensorProto::INT64);
      TensorShapeProto output_shape;
      auto dim = output_shape.add_dim();
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& input_shape = getInputShape(ctx, 0);
        dim->set_dim_value(input_shape.dim_size());
      }
      output_shape.add_dim();
      updateOutputShape(ctx, 0, output_shape);
    }));

static constexpr const char* ReverseSequence_ver10_doc = R"DOC(
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ReverseSequence,
    10,
    OpSchema().FillUsing(ReverseSequence_v10_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 2)) {
        return;
      }

      auto& first_input_shape = getInputShape(ctx, 0);
      if (first_input_shape.dim_size() < 2) {
        fail_shape_inference("'input' must have rank >= 2");
      }
      auto& seq_len_input_shape = getInputShape(ctx, 1);
      if (seq_len_input_shape.dim_size() != 1) {
        fail_shape_inference("'sequence_lens' must have rank of 1");
      }

      propagateShapeFromInputToOutput(ctx, 0, 0);
    }));

static constexpr const char* Unique_ver11_doc = R"DOC(
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
```
input_X = [2, 1, 1, 3, 4, 3]
attribute_sorted = 0
attribute_axis = None
output_Y = [2, 1, 3, 4]
output_indices = [0, 1, 3, 4]
output_inverse_indices = [0, 1, 1, 2, 3, 2]
output_counts = [1, 2, 2, 1]
```

Example 2:
```
input_X = [[1, 3], [2, 3]]
attribute_sorted = 1
attribute_axis = None
output_Y = [1, 2, 3]
output_indices = [0, 2, 1]
output_inverse_indices = [0, 2, 1, 2]
output_counts = [1, 1, 2]
```

Example 3:
```
input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
attribute_sorted = 1
attribute_axis = 0
output_Y = [[1, 0, 0], [2, 3, 4]]
output_indices = [0, 2]
output_inverse_indices = [0, 0, 1]
output_counts = [2, 1]
```

Example 4:
```
input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
            [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
attribute_sorted = 1
attribute_axis = 1
```

intermediate data are presented below for better understanding:
there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
```
A: [[1, 1], [1, 1]],
   [[0, 1], [0, 1]],
   [[2, 1], [2, 1]],
   [[0, 1], [0, 1]].
```

there are 3 unique subtensors:
```
[[1, 1], [1, 1]],
[[0, 1], [0, 1]],
[[2, 1], [2, 1]].
```

sorted unique subtensors:
```
B: [[0, 1], [0, 1]],
   [[1, 1], [1, 1]],
   [[2, 1], [2, 1]].
```

output_Y is constructed from B:
```
[[[0. 1.], [1. 1.], [2. 1.]],
 [[0. 1.], [1. 1.], [2. 1.]]]
```

output_indices is to map from B to A:
```
[1, 0, 2]
```

output_inverse_indices is to map from A to B:
```
[1, 0, 2, 0]
```

output_counts:
```
[2, 1, 1]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    OpSchema().FillUsing(Unique_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      const TypeProto* xTensorProto = ctx.getInputType(0);
      TypeProto* yTensorProto = ctx.getOutputType(0);
      TypeProto* indicesTensorProto = nullptr;
      TypeProto* inverseIndicesTensorProto = nullptr;
      TypeProto* countsTensorProto = nullptr;

      // 'indices', 'inverse_indices', and 'counts' are 1-D tensors of
      // unknown dimension.
      // Shape inference will happen even in case of empty optional outputs,
      // graph-level shape inference should not propagate the shape downstream for empty optional outputs.
      auto num_outputs = ctx.getNumOutputs();
      if (num_outputs >= 2) {
        indicesTensorProto = ctx.getOutputType(1);
        updateOutputElemType(ctx, 1, TensorProto::INT64);
        indicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      }

      if (num_outputs >= 3) {
        inverseIndicesTensorProto = ctx.getOutputType(2);
        updateOutputElemType(ctx, 2, TensorProto::INT64);
        inverseIndicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      }

      if (num_outputs >= 4) {
        countsTensorProto = ctx.getOutputType(3);
        updateOutputElemType(ctx, 3, TensorProto::INT64);
        countsTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      }

      auto axisAttr = ctx.getAttribute("axis");
      if (!axisAttr) {
        // 'axis' is not provided. Input 'X' is flattened.
        // 'Y' is a 1-D tensor of unknown dimension.
        yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      } else {
        // 'axis' is provided.
        int axis = static_cast<int>(axisAttr->i());
        if (!xTensorProto->tensor_type().has_shape()) {
          return;
        }
        const TensorShapeProto& input_shape = xTensorProto->tensor_type().shape();
        int rank = input_shape.dim_size();
        if (axis < 0)
          axis += rank;
        if (axis < 0 || axis >= rank) {
          fail_shape_inference("Invalid value for attribute axis");
        }
        // 'Y' has the same shape as 'X' except in the 'axis' dimension
        // which is unknown.
        for (int i = 0; i < input_shape.dim_size(); i++) {
          auto dim = yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          if (i != axis) {
            *dim = input_shape.dim(i);
          }
        }
      }
    }));

static constexpr const char* GatherND_ver13_doc = R"DOC(
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

**Example 1**

```
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]
```

**Example 2**

```
batch_dims = 0
data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
indices = [[1],[0]]      # indices_shape = [2, 1]
output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
```

**Example 3**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

**Example 4**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
```

**Example 5**

```
batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    13,
    OpSchema().FillUsing(GatherND_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      if (!hasNInputShapes(ctx, 2)) {
        // cannot proceed with shape or rank inference
        return;
      }

      const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
      const auto data_rank = data_shape.dim_size();

      const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
      const auto indices_rank = indices_shape.dim_size();

      int64_t batch_dims_data = getAttribute(ctx, "batch_dims", 0);
      if (data_rank < 1 || indices_rank < 1) {
        fail_shape_inference(
            "Both `data` and `indices` input tensors in GatherND op "
            "need to have rank larger than 0.");
      }

      // cannot ascertain if the input shapes are valid if shape of
      // `indices` is missing last dimension value so return at this point
      if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
        return;
      }

      const auto last_index_dimension = indices_shape.dim(indices_rank - 1).dim_value() + batch_dims_data;

      if (last_index_dimension > data_rank) {
        fail_shape_inference(
            "Last dimension of `indices` input tensor in GatherND op "
            "must not be larger than the rank of `data` tensor");
      }

      for (int i = 0; i < indices_rank - 1; ++i) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
      }

      for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    25,
    OpSchema().FillUsing(Pad_v25_FillSpec).TypeAndShapeInferenceFunction(padShapeInference));

static constexpr const char* Trilu_ver14_doc = R"DOC(
Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute "upper" determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the "upper" attribute is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Trilu,
    14,
    OpSchema().FillUsing(Trilu_v14_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      // Shape inference needs the input data shape
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        const int rank = static_cast<int>(input_shape.dim_size());
        if (rank < 2) {
          fail_shape_inference("Input rank must be >= 2.")
        }
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

static constexpr const char* CenterCropPad_ver18_doc = R"DOC(
Center crop or pad an input to given dimensions.

The crop/pad dimensions can be specified for a subset of the `axes`; unspecified dimensions will remain unchanged.

If the input dimensions are larger than the target crop dimensions, a centered cropping window will be extracted
from the input. The starting value for the cropping window is rounded down, which means that if the difference
between the input shape and the crop shape is odd, the cropping window will be shifted half a pixel to the left
of the input center.

If the input dimensions are smaller than the target crop dimensions, the input will be padded equally on both sides
to center it in the output. In cases where the total number of padding pixels is odd, an additional pixel will be
added to the right side.

The padding value used is zero.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    CenterCropPad,
    18,
    OpSchema()
        .FillUsing(CenterCropPad_v18_FillSpec)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumInputs() != 2) {
            fail_type_inference("CenterCropPad op must have 2 inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if shape is initializer
          const TensorProto* cropShapeInitializer = ctx.getInputData(1);
          if (!cropShapeInitializer) {
            return;
          }

          // don't know data_type - can't proceed
          if (!cropShapeInitializer->has_data_type())
            return;

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const int64_t input_rank = input_shape.dim_size();

          std::vector<int64_t> shape;
          if (cropShapeInitializer->data_type() == TensorProto::INT64) {
            const auto data = ParseData<int64_t>(cropShapeInitializer);
            shape.insert(shape.end(), data.begin(), data.end());
          } else if (cropShapeInitializer->data_type() == TensorProto::INT32) {
            const auto data = ParseData<int32_t>(cropShapeInitializer);
            shape.insert(shape.end(), data.begin(), data.end());
          } else {
            // unaccepted data type
            fail_shape_inference("`shape` only supports `int32_t` or `int64_t` inputs");
          }

          auto axes_attr = ctx.getAttribute("axes");
          std::vector<int64_t> axes;
          if (axes_attr) {
            axes = RetrieveValues<int64_t>(*axes_attr);
            checkAxesRange(axes, input_rank);
            adjustNegativeAxes(axes, input_rank);
            checkDuplicateAxes(axes, input_rank);
          } else {
            axes.resize(input_rank);
            std::iota(axes.begin(), axes.end(), 0);
          }

          if (shape.size() != axes.size()) {
            fail_shape_inference(
                "Number of elements of input 'shape' (",
                shape.size(),
                ") does not match the number of axes (",
                axes.size(),
                ").");
          }

          // Populating default dims
          std::vector<TensorShapeProto_Dimension*> out_dims(input_rank);
          auto output_shape = getOutputShape(ctx, 0);
          for (int i = 0; i < input_rank; ++i) {
            out_dims[i] = output_shape->add_dim();
            const auto& input_dim = input_shape.dim(i);
            if (input_dim.has_dim_value()) {
              out_dims[i]->set_dim_value(input_dim.dim_value());
            } else if (input_dim.has_dim_param()) {
              out_dims[i]->set_dim_param(input_dim.dim_param());
            }
          }
          int j = 0;
          for (int axis : axes) {
            out_dims[axis]->set_dim_value(shape[j++]);
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          FunctionBuilder builder(functionProto);
          builder.Const("k2", std::vector<int64_t>{2});

          auto axes_attr = ctx.getAttribute("axes");
          if (axes_attr) { // axes provided, need to work on a subset of dimensions
            builder.Add("axes_input = Constant <value_ints : ints = @axes>()");
            builder.Add("x_shape_alldims = Shape (input_data)").Add("x_shape = Gather (x_shape_alldims, axes_input)");
          } else { // axes not provided, assuming all dims
            builder.Add("x_shape = Shape (input_data)");
          }

          // First: Pad step
          builder.Add("padded_sh = Max(x_shape, shape)")
              .Add("pad_amount = Sub(padded_sh, x_shape)")
              .Add("pad_amount_left = Div(pad_amount, k2)")
              .Add("pad_amount_right = Sub(pad_amount, pad_amount_left)")
              .Add("pads = Concat <axis = 0> (pad_amount_left, pad_amount_right)");
          if (axes_attr)
            builder.Add("padded_input = Pad (input_data, pads, , axes_input)");
          else
            builder.Add("padded_input = Pad (input_data, pads)");

          // Second: Slice step
          if (axes_attr) {
            builder.Add("x_shape_alldims2 = Shape (padded_input)")
                .Add("x_shape2 = Gather (x_shape_alldims2, axes_input)");
          } else {
            builder.Add("x_shape2 = Shape (padded_input)");
          }

          builder.Add("sh_diff = Sub (x_shape2, shape)")
              .Add("start_dims = Div (sh_diff, k2)")
              .Add("end_dims = Add (start_dims, shape)");
          if (axes_attr)
            builder.Add("output_data = Slice (padded_input, start_dims, end_dims, axes_input)");
          else
            builder.Add("output_data = Slice (padded_input, start_dims, end_dims)");

          schema.BuildFunction(functionProto);
          return true;
        }));

static constexpr const char* TensorScatter_ver24_doc = R"DOC(
TensorScatter is a generic tensor update operation, motivated by the requirements for KV cache updates for Attention
ops commonly found in LLMs. It is a functional operation that models an in-place update to a KV cache buffer.

The past and present cache tensors have the same shape (batch_size, D1, D2, ..., max_sequence_length, ..., Dn), with
the sequence dimension (indicated by the `axis` attribute) being max_sequence_length, so the sizes of these tensors do
not need to grow between iterations. The `update` tensor's shape only differs from the cache tensors in the sequence
dimension: (batch_size, D1, D2, ..., sequence_length, ..., Dn), where sequence_length <= max_sequence_length.

The optional `write_indices` input indicates the write index for each sample in the batch, assumed to be zero
if not provided. When the `mode` attribute is set to "circular", the write index is modulo max_sequence_length.
The operation can be described using the following pseudocode:

```
for prefix_idx in np.ndindex(past_cache.shape[:axis]):
    batch_idx = prefix_idx[0]
    for sequence_idx in range(sequence_length):
        cache_idx = (*prefix_idx, write_indices[batch_idx] + sequence_idx)
        if mode == "circular":
            cache_idx = tuple(np.mod(np.asarray(cache_idx), max_sequence_length))
        update_idx = (*prefix_idx, sequence_idx)
        present_cache[cache_idx] = update[update_idx]
```

During the prefill phase of attention, only the first two inputs are needed. During the decode phase, `write_indices`
is also needed so that the incoming key or value update can be appended after the last valid token for each sample
in the batch.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TensorScatter,
    24,
    OpSchema().FillUsing(TensorScatter_v24_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Check that updates and past_cache have the same rank
      if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
        const auto& past_cache_shape = getInputShape(ctx, 0);
        const auto& update_shape = getInputShape(ctx, 1);
        const int past_cache_rank = static_cast<int>(past_cache_shape.dim_size());
        const int update_rank = static_cast<int>(update_shape.dim_size());

        if (past_cache_rank != update_rank) {
          fail_shape_inference(
              "past_cache and update must have the same rank. past_cache_rank=",
              past_cache_rank,
              ", update_rank=",
              update_rank);
        }

        // Check that all dimensions match except the axis dimension
        // Read axis from attribute, default is -2
        auto axis_attr = ctx.getAttribute("axis");
        int64_t axis = axis_attr ? axis_attr->i() : -2;

        // Handle negative axis
        if (axis < 0) {
          axis += past_cache_rank;
        }

        // Validate axis is within bounds
        if (axis <= 0 || axis >= past_cache_rank) {
          fail_shape_inference("axis ", axis, " is out of range for rank ", past_cache_rank);
        }
        for (int i = 0; i < past_cache_rank; ++i) {
          if (i != axis) {
            // All dimensions except axis should match
            Dim dim;
            unifyInputDim(ctx, 0, i, dim);
            unifyInputDim(ctx, 1, i, dim);
          }
        }
      }

      // Check that write_indices is of shape (batch_size,)
      if (hasInputShape(ctx, 2)) {
        checkInputRank(ctx, 2, 1);
        Dim batch_size;
        unifyInputDim(ctx, 0, 0, batch_size);
        unifyInputDim(ctx, 2, 0, batch_size);
      }

      if (hasNInputShapes(ctx, 1)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }
    }));

} // namespace ONNX_NAMESPACE
