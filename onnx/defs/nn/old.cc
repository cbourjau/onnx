// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

// For GlobalPool operations.
static void globalPoolTypeShapeInference_opset2(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // (N, C, 1, 1, ..., 1)
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

static std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator_opset2(const char* op_type, const char* op) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
        ReplaceAll(doc, "{op_type}", op_type);
        ReplaceAll(doc, "{op}", op););
    schema.SetDoc(doc);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(globalPoolTypeShapeInference_opset2);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    2,
    OpSchema().FillUsing(GlobalLpPool_v2_FillSpec).TypeAndShapeInferenceFunction(globalPoolTypeShapeInference));

constexpr const char* pads_doc_opset11 =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
constexpr const char* conv_auto_pad_doc_opset19 =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";
constexpr const char* conv_transpose_auto_pad_doc_opset19 =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = input_shape[i] * strides[i]` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";

static void convPoolShapeInference_opset19(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // we need the first input shape for this inference.
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides = defs::nn::utils::getConvPoolStrides(ctx, n_input_dims);

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape = ctx.getInputType(input2Idx)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      auto input_dims_size = n_input_dims;
      for (size_t i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape.dim(2 + i).has_dim_value()) {
            continue;
          }
          residual = input_shape.dim(2 + i).dim_value();
          while (residual >= stride) {
            residual -= stride;
          }
        }
        if (i >= effective_kernel_shape.size()) {
          fail_shape_inference("kernel shape should have ", input_dims_size, " values in ", ctx.getDisplayName(), ".");
        }
        int64_t total_pad = residual == 0 ? effective_kernel_shape[i] - stride : effective_kernel_shape[i] - residual;
        total_pad = std::max<int64_t>(total_pad, 0);
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(1);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    const auto& second_input_shape = getInputShape(ctx, input2Idx);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto* newdim = output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }
    // how big is the input, including padding
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions = 0;

    if (ceil_mode == 1)
      strided_kernel_positions = static_cast<int64_t>(
          std::ceil((effective_input_size - effective_kernel_shape[i]) / static_cast<float>(strides[i])));
    else
      strided_kernel_positions = (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    newdim->set_dim_value(1 + strided_kernel_positions);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    13,
    OpSchema().FillUsing(Dropout_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }

      if (ctx.getNumInputs() > 1 && hasInputShape(ctx, 1)) {
        auto& ratio_input_shape = getInputShape(ctx, 1);
        if (static_cast<int>(ratio_input_shape.dim_size()) != 0) {
          fail_shape_inference("Ratio of Dropout must be a scalar.");
        }
      }

      if (ctx.getNumInputs() > 2 && hasInputShape(ctx, 2)) {
        auto& training_mode_input_shape = getInputShape(ctx, 2);
        if (static_cast<int>(training_mode_input_shape.dim_size()) != 0) {
          fail_shape_inference("training_mode of Dropout must be a scalar.");
        }
      }

      if (ctx.getNumOutputs() == 2) {
        updateOutputElemType(ctx, 1, TensorProto::BOOL);
        if (hasNInputShapes(ctx, 1)) {
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    1,
    OpSchema().FillUsing(LpNormalization_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    6,
    OpSchema().FillUsing(InstanceNormalization_v6_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

static void maxUnpoolShapeInference_opset11(InferenceContext& ctx) {
  // we need at least two inputs to have a shape for this inference.
  if (ctx.getNumInputs() != 2 && ctx.getNumInputs() != 3) {
    fail_type_inference("MaxUnpool op must have either two or three inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    return; // If first input does not have shape, we cannot infer much.
  }
  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor X must have at least 2 dimensions.");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size.");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size.");
    }
  } else {
    fail_shape_inference("Attribute kernel_shape must be specified.");
  }

  if (ctx.getNumInputs() == 3) {
    // If the third input, output_size, is specified, then use that instead
    // of inferring shape from inputs.
    if (hasInputShape(ctx, 2)) {
      const auto& output_shape = getInputShape(ctx, 2);
      if (output_shape.dim_size() != 1) {
        fail_type_inference("'output_shape' must be rank 1 tensor.");
      }
      if (output_shape.dim(0).has_dim_value() &&
          static_cast<int>(output_shape.dim(0).dim_value()) != input_shape.dim_size()) {
        fail_shape_inference("'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be
            // determined at runtime.
  }

  auto* final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1); // channels should be the second dim of second input.

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto* newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value = strides[i] * (input_shape.dim(2 + i).dim_value() - 1);
    newdim_value += kernel_shape[i];
    newdim_value -= pads[i];
    newdim_value -= pads[i + kernel_shape_size];

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

// For GlobalPool operations.
static void globalPoolTypeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // (N, C, 1, 1, ..., 1)
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

static std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator_opset1(const char* op_type, const char* op) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
        ReplaceAll(doc, "{op_type}", op_type);
        ReplaceAll(doc, "{op}", op););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(globalPoolTypeShapeInference_opset1);
  };
}
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    1,
    OpSchema().FillUsing(GlobalAveragePool_v1_FillSpec).TypeAndShapeInferenceFunction(globalPoolTypeShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    GlobalMaxPool,
    1,
    OpSchema().FillUsing(GlobalMaxPool_v1_FillSpec).TypeAndShapeInferenceFunction(globalPoolTypeShapeInference));

static void convTransposeShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return; // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
    if (kernel_shape.size() != n_input_dims) {
      return; // weight rank mismatch
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if (nullptr != auto_pad_attr && auto_pad_attr->s() != "NOTSET") {
      fail_shape_inference("The pads attribute cannot be used simultaneously with auto_pad attribute");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t total_pad = effective_kernel_shape[i] - strides[i];
        total_pad = std::max<int64_t>(total_pad, 0);
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) { // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto* final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) * group; // channels should be the second dim of second input
                                                                 // multiply group.

  int size_of_output = 0;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO(ONNX): throw exception?
          return; // output shape value cannot be smaller than the input shape
                  // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim = strides[i] * (input_shape.dim(i + 2).dim_value() - 1) + output_padding[i] +
            effective_kernel_shape[i] - pads[i] - pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    19,
    OpSchema().FillUsing(DeformConv_v19_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference_opset19(ctx, true, false, 0, 1);
    }));

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator_opset11(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )DOC";
        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; has size (N x C x H x W)"
        ", where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (C x M/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "weight shape will be (C x M/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... x kn) is the dimension of the kernel. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the convolution. The "
        "output dimensions are functions of the kernel size, stride size, "
        "pad lengths and group count. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_shape",
        "The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified "
        "pads values are ignored. See doc for details for equations to generate pads. Note that the output_shape attribute value "
        "should not include dimensions for batch size and channels, which are automatically inferred.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_padding",
        "Additional elements added to the side with higher coordinate indices in the output. "
        "Each padding value in \"output_padding\" must be less than the corresponding stride/dilation dimension. "
        "By default, this attribute is a zero vector. "
        "Note that this attribute doesn't directly affect the computed output values. "
        "It only controls the selection of the computed values, "
        "so changing this attribute only adds or removes output elements. "
        "If \"output_shape\" is explicitly provided, "
        "\"output_padding\" does not contribute additional size to \"output_shape\" but "
        "participates in the computation of the needed padding amount. "
        "This is also called adjs or adjustment in some frameworks.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_transpose_auto_pad_doc_opset19, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc_opset11, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction(convTransposeShapeInference_opset11);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    11,
    OpSchema()
        .FillUsing(ConvTranspose_v11_FillSpec)
        .TypeAndShapeInferenceFunction(convTransposeShapeInference_opset11));

static void convShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference_opset19(ctx, true, false, 0, 1);
}

static std::function<void(OpSchema&)> ConvOpSchemaGenerator_opset11(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
The convolution operator consumes an input tensor and {filter_desc}, and
computes the output.)DOC";
        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; "
        "has size (N x C x H x W), where N is the batch size, "
        "C is the number of channels, and H and W are the "
        "height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
        "Optionally, if dimension denotation is "
        "in effect, the operation expects input data tensor "
        "to arrive with the dimension denotation of [DATA_BATCH, "
        "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (M x C/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... kn) is the dimension of the kernel. "
        "Optionally, if dimension denotation is in effect, "
        "the operation expects the weight tensor to arrive "
        "with the dimension denotation of [FILTER_OUT_CHANNEL, "
        "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
        "Assuming zero based indices for the shape array, "
        "X.shape[1] == (W.shape[1] * group) == C and "
        "W.shape[0] mod G == 0. Or in other words "
        "FILTER_IN_CHANNEL multiplied by the number of groups "
        "should be equal to DATA_CHANNEL and the number of "
        "feature maps M should be a multiple of the number of "
        "groups G.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the "
        "convolution. The output dimensions are functions "
        "of the kernel size, stride size, and pad lengths.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc_opset19, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc_opset11, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction(convShapeInference_opset11);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Conv,
    11,
    OpSchema().FillUsing(Conv_v11_FillSpec).TypeAndShapeInferenceFunction(convShapeInference_opset11));

static void roiPoolTypeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // rois is the second input.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto rios_shape = ctx.getInputType(1)->tensor_type().shape();

  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }
  if (rios_shape.dim_size() != 2) {
    fail_shape_inference("RoIs tensor must have 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pooled_shape;
  if (getRepeatedAttribute(ctx, "pooled_shape", pooled_shape)) {
    if (pooled_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute pooled_shape has incorrect length");
    }
  } else {
    fail_shape_inference("Attribute pooled_shape must be specified");
  }

  // (num_rois, channels, pooled_shape[0], pooled_shape[1])
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *output_shape->add_dim() = rios_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);
  output_shape->add_dim()->set_dim_value(pooled_shape[0]);
  output_shape->add_dim()->set_dim_value(pooled_shape[1]);
}

static std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator_opset1(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 ROI {name} pool consumes an input tensor X and region of interests (RoIs) to
 apply {name} pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("pooled_shape", "ROI pool output shape (height, width).", AttributeProto::INTS);
    schema.Attr(
        "spatial_scale",
        "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.",
        AttributeProto::FLOAT,
        1.f);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "rois",
        "RoIs (Regions of Interest) to pool over. Should "
        "be a 2-D tensor of shape (num_rois, 5) given as "
        "[[batch_id, x1, y1, x2, y2], ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(
        0,
        "Y",
        "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(roiPoolTypeShapeInference_opset1);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    MaxRoiPool,
    1,
    OpSchema().FillUsing(MaxRoiPool_v1_FillSpec).TypeAndShapeInferenceFunction(roiPoolTypeShapeInference_opset1));

static void lpPoolShapeInference_opset18(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference_opset19(ctx, true, true, 0, 1);
}

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator_opset18(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc_opset19, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc_opset11, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the "
        "batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(lpPoolShapeInference_opset18);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    18,
    OpSchema().FillUsing(LpPool_v18_FillSpec).TypeAndShapeInferenceFunction(lpPoolShapeInference_opset18));

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    11,
    OpSchema().FillUsing(MaxUnpool_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      maxUnpoolShapeInference_opset11(ctx);
    }));

static std::vector<std::string> GetSupportedDataTypesForPoolingOps_opset19(bool supports8bit) {
  if (supports8bit) {
    return {types::Float16, types::Float, types::Double, types::Int8, types::UInt8};
  }
  return {types::Float16, types::Float, types::Double};
}

static std::function<void(InferenceContext&)> poolShapeInferenceFunc_opset19(bool use_dilation) {
  return [use_dilation](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getNumOutputs() > 1) {
      // MaxPool with two outputs case.
      auto* output_type = ctx.getOutputType(1);
      if (output_type->value_case() == TypeProto::kTensorType ||
          output_type->value_case() == TypeProto::VALUE_NOT_SET) {
        output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
      }
    }
    convPoolShapeInference_opset19(ctx, use_dilation, true, 0, 1);
  };
}

static void poolShapeInference_opset19_with_dilation(InferenceContext& ctx) {
  poolShapeInferenceFunc_opset19(true)(ctx);
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator_opset19(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool use_dilation,
    bool supports8bit = false) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{opName}", opName);
        ReplaceAll(doc, "{additionalDescription}", additionalDescription);
        ReplaceAll(
            doc,
            "{kernelSpatialShape}",
            use_dilation ? "((kernel_spatial_shape[i] - 1) * dilations[i] + 1)" : "kernel_spatial_shape[i]"););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc_opset19, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc_opset11, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForPoolingOps_opset19(supports8bit),
        supports8bit ? "Constrain input and output types to float and 8 bit tensors."
                     : "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(poolShapeInference_opset19_with_dilation);
  };
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator_opset19_with_dilation(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool supports8bit = false) {
  return PoolOpSchemaGenerator_opset19(name, opName, additionalDescription, true, supports8bit);
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    19,
    OpSchema()
        .FillUsing(AveragePool_v19_FillSpec)
        .TypeAndShapeInferenceFunction(poolShapeInference_opset19_with_dilation));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    12,
    OpSchema().FillUsing(MaxPool_v12_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_opset19_with_dilation));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    12,
    OpSchema().FillUsing(Dropout_v12_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
      }

      if (ctx.getNumInputs() > 1 && hasInputShape(ctx, 1)) {
        auto& ratio_input_shape = getInputShape(ctx, 1);
        if (static_cast<int>(ratio_input_shape.dim_size()) != 0) {
          fail_shape_inference("Ratio of Dropout must be a scalar.");
        }
      }

      if (ctx.getNumInputs() > 2 && hasInputShape(ctx, 2)) {
        auto& training_mode_input_shape = getInputShape(ctx, 2);
        if (static_cast<int>(training_mode_input_shape.dim_size()) != 0) {
          fail_shape_inference("training_mode of Dropout must be a scalar.");
        }
      }

      if (ctx.getNumOutputs() == 2) {
        updateOutputElemType(ctx, 1, TensorProto::BOOL);
        if (hasNInputShapes(ctx, 1)) {
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    24,
    OpSchema().FillUsing(Flatten_v24_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < 0) {
        axis += rank;
      }
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    23,
    OpSchema().FillUsing(Flatten_v23_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < 0) {
        axis += rank;
      }
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    21,
    OpSchema().FillUsing(Flatten_v21_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < 0) {
        axis += rank;
      }
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    13,
    OpSchema().FillUsing(Flatten_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < 0) {
        axis += rank;
      }
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    11,
    OpSchema().FillUsing(Flatten_v11_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < 0) {
        axis += rank;
      }
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

static constexpr const char* LRN_ver1_doc = R"DOC(
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    1,
    OpSchema().FillUsing(LRN_v1_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static constexpr const char* mvn_ver9_doc = R"DOC(
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
)DOC";

static const std::vector<int64_t> old_mvn_default_axes = {0, 2, 3};

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    9,
    OpSchema()
        .FillUsing(MeanVarianceNormalization_v9_FillSpec)
        .FunctionBody(
            FunctionBodyHelper::BuildNodes(
                {// nodes: {outputs, op, inputs, attributes}
                 FunctionBodyHelper::Const<float>("Exponent", 2.0f),
                 FunctionBodyHelper::Const<float>("Epsilon", static_cast<float>(1e-9)),
                 {{"X_RM"}, "ReduceMean", {"X"}, {MakeRefAttribute("axes", AttributeProto::INTS)}},
                 {{"EX_squared"}, "Pow", {"X_RM", "Exponent"}},
                 {{"X_squared"}, "Pow", {"X", "Exponent"}},
                 {{"E_Xsquared"}, "ReduceMean", {"X_squared"}, {MakeRefAttribute("axes", AttributeProto::INTS)}},
                 {{"Variance"}, "Sub", {"E_Xsquared", "EX_squared"}},
                 {{"STD"}, "Sqrt", {"Variance"}},
                 {{"X_variance"}, "Sub", {"X", "X_RM"}},
                 {{"Processed_STD"}, "Add", {"STD", "Epsilon"}},
                 {{"Y"}, "Div", {"X_variance", "Processed_STD"}}})));

constexpr const char* pads_doc2 =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
constexpr const char* auto_pad_doc2 =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding.";
constexpr const char* auto_pad_doc3 =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";

static void convPoolShapeInference_opset1_to_11(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // we need the first input shape for this inference.
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides = defs::nn::utils::getConvPoolStrides(ctx, n_input_dims);

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape = ctx.getInputType(input2Idx)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape.dim(2 + i).has_dim_value()) {
            continue;
          }
          residual = input_shape.dim(2 + i).dim_value();
          while (residual >= stride) {
            residual -= stride;
          }
        }
        int64_t total_pad = residual == 0 ? effective_kernel_shape[i] - stride : effective_kernel_shape[i] - residual;
        total_pad = std::max<int64_t>(total_pad, 0);
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(1);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    const auto& second_input_shape = getInputShape(ctx, input2Idx);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto* newdim = output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }
    // how big is the input, including padding
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions = 0;

    if (ceil_mode == 1)
      strided_kernel_positions = static_cast<int64_t>(
          std::ceil((effective_input_size - effective_kernel_shape[i]) / static_cast<float>(strides[i])));
    else
      strided_kernel_positions = (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    newdim->set_dim_value(1 + strided_kernel_positions);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

static void poolShapeInference_opset1_to_8(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* output_type = ctx.getOutputType(1);
    if (output_type->value_case() == TypeProto::kTensorType || output_type->value_case() == TypeProto::VALUE_NOT_SET) {
      output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    }
  }
  convPoolShapeInference_opset1_to_11(ctx, false, true, 0, 1);
}

static std::function<void(OpSchema&)>
PoolOpSchemaGenerator_opset1_to_8(const char* name, const char* opName, const char* additionalDescription) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{opName}", opName);
        ReplaceAll(doc, "{additionalDescription}", additionalDescription););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr("strides", "Stride along each spatial axis.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc2, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T");
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_8);
  };
}

static std::function<void(InferenceContext&)> poolShapeInferenceFunc_opset1_to_11(bool use_dilation) {
  return [use_dilation](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (ctx.getNumOutputs() > 1) {
      // MaxPool with two outputs case.
      auto* output_type = ctx.getOutputType(1);
      if (output_type->value_case() == TypeProto::kTensorType ||
          output_type->value_case() == TypeProto::VALUE_NOT_SET) {
        output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
      }
    }
    convPoolShapeInference_opset1_to_11(ctx, use_dilation, true, 0, 1);
  };
}

static void poolShapeInference_opset1_to_11_no_dilation(InferenceContext& ctx) {
  poolShapeInferenceFunc_opset1_to_11(false)(ctx);
}

static void poolShapeInference_opset1_to_11_with_dilation(InferenceContext& ctx) {
  poolShapeInferenceFunc_opset1_to_11(true)(ctx);
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator_opset10_to_11(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool use_dilation,
    int opsetNum) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{opName}", opName);
        ReplaceAll(doc, "{additionalDescription}", additionalDescription);
        ReplaceAll(
            doc,
            "{kernelSpatialShape}",
            use_dilation ? "((kernel_spatial_shape[i] - 1) * dilations[i] + 1)" : "kernel_spatial_shape[i]"););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        opsetNum == 11
            ? "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis."
            : "Stride along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc2, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T");
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(
        use_dilation ? poolShapeInference_opset1_to_11_with_dilation : poolShapeInference_opset1_to_11_no_dilation);
  };
}

static std::vector<std::string> GetSupportedDataTypesForPoolingOps_opset11(bool supports8bit) {
  if (supports8bit) {
    return {types::Float16, types::Float, types::Double, types::Int8, types::UInt8};
  }
  return {types::Float16, types::Float, types::Double};
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator_opset11(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool use_dilation,
    bool supports8bit = false) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
or when ceil_mode is disabled:
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
 ```

 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{opName}", opName);
        ReplaceAll(doc, "{additionalDescription}", additionalDescription);
        ReplaceAll(
            doc,
            "{kernelSpatialShape}",
            use_dilation ? "((kernel_spatial_shape[i] - 1) * dilations[i] + 1)" : "kernel_spatial_shape[i]"););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc3, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForPoolingOps_opset11(supports8bit),
        supports8bit ? "Constrain input and output types to float and 8 bit tensors."
                     : "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(
        use_dilation ? poolShapeInference_opset1_to_11_with_dilation : poolShapeInference_opset1_to_11_no_dilation);
  };
}

static std::function<void(OpSchema&)>
PoolOpSchemaGenerator_opset10_no_dilation(const char* name, const char* opName, const char* additionalDescription) {
  return PoolOpSchemaGenerator_opset10_to_11(name, opName, additionalDescription, false, 10);
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator_opset11_with_dilation(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool supports8bit = false) {
  return PoolOpSchemaGenerator_opset11(name, opName, additionalDescription, true, supports8bit);
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    1,
    OpSchema().FillUsing(AveragePool_v1_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_8));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    7,
    OpSchema().FillUsing(AveragePool_v7_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_8));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    10,
    OpSchema()
        .FillUsing(AveragePool_v10_FillSpec)
        .TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_11_no_dilation));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    11,
    OpSchema()
        .FillUsing(AveragePool_v11_FillSpec)
        .TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_11_with_dilation));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    1,
    OpSchema().FillUsing(MaxPool_v1_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_8));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    8,
    OpSchema().FillUsing(MaxPool_v8_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_8));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    10,
    OpSchema()
        .FillUsing(MaxPool_v10_FillSpec)
        .TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_11_with_dilation));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    11,
    OpSchema()
        .FillUsing(MaxPool_v11_FillSpec)
        .TypeAndShapeInferenceFunction(poolShapeInference_opset1_to_11_with_dilation));

static void maxUnpoolShapeInference_opset9(InferenceContext& ctx) {
  // we need at least two inputs to have a shape for this inference.
  if (ctx.getNumInputs() != 2 && ctx.getNumInputs() != 3) {
    fail_type_inference("MaxUnpool op must have either two or three inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    return; // If first input does not have shape, we cannot infer much.
  }
  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor X must have at least 2 dimensions.");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size.");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size.");
    }
  } else {
    fail_shape_inference("Attribute kernel_shape must be specified.");
  }

  if (ctx.getNumInputs() == 3) {
    // If the third input, output_size, is specified, then use that instead
    // of inferring shape from inputs.
    if (hasInputShape(ctx, 2)) {
      const auto& output_shape = getInputShape(ctx, 2);
      if (output_shape.dim_size() != 1) {
        fail_type_inference("'output_shape' must be rank 1 tensor.");
      }
      if (output_shape.dim(0).has_dim_value() &&
          static_cast<int>(output_shape.dim(0).dim_value()) != input_shape.dim_size()) {
        fail_shape_inference("'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be
            // determined at runtime.
  }

  auto* final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1); // channels should be the second dim of second input.

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto* newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value = strides[i] * (input_shape.dim(2 + i).dim_value() - 1);
    newdim_value += kernel_shape[i];
    newdim_value -= pads[i];
    newdim_value -= pads[i + kernel_shape_size];

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    9,
    OpSchema().FillUsing(MaxUnpool_v9_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      maxUnpoolShapeInference_opset9(ctx);
    }));

constexpr const char* pads_doc1 =
    "Padding for the beginning and ending along each axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute.";
constexpr const char* auto_pad_doc1 =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
    "only intended to support legacy uses, and for framework authors, one is explicitly "
    "encouraged to use explicit padding specified in the pads attribute.";

constexpr const char* LpPool_ver1_doc = R"DOC(
 LpPool consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";

ONNX_OPERATOR_SET_SCHEMA(LpPool, 1, OpSchema().FillUsing(LpPool_v1_FillSpec));

static void lpPoolShapeInference_opset2(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference_opset1_to_11(ctx, false, true, 0, 1);
}

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator_opset2(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr("strides", "Stride along each spatial axis.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc2, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the "
        "batch size.",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T");
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(lpPoolShapeInference_opset2);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    2,
    OpSchema().FillUsing(LpPool_v2_FillSpec).TypeAndShapeInferenceFunction(lpPoolShapeInference_opset2));

static constexpr const char* GlobalLpPool_ver1_doc = R"DOC(
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.)DOC";

static void lpPoolShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference_opset1_to_11(ctx, false, true, 0, 1);
}

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator_opset11(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc3, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the "
        "batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(lpPoolShapeInference_opset11);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    11,
    OpSchema().FillUsing(LpPool_v11_FillSpec).TypeAndShapeInferenceFunction(lpPoolShapeInference_opset11));

static void convShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference_opset1_to_11(ctx, true, false, 0, 1);
}

static std::function<void(OpSchema&)> ConvOpSchemaGenerator_opset1(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
The convolution operator consumes an input tensor and {filter_desc}, and
computes the output.)DOC";
        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; "
        "has size (N x C x H x W), where N is the batch size, "
        "C is the number of channels, and H and W are the "
        "height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
        "Optionally, if dimension denotation is "
        "in effect, the operation expects input data tensor "
        "to arrive with the dimension denotation of [DATA_BATCH, "
        "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T");
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (M x C/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... kn) is the dimension of the kernel. "
        "Optionally, if dimension denotation is in effect, "
        "the operation expects the weight tensor to arrive "
        "with the dimension denotation of [FILTER_OUT_CHANNEL, "
        "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
        "X.shape[1] == (W.shape[1] * group) == C "
        "(assuming zero based indices for the shape array). "
        "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
        "T");
    schema.Input(2, "B", "Optional 1D bias to be added to the convolution, has size of M.", "T", OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the "
        "convolution. The output dimensions are functions "
        "of the kernel size, stride size, and pad lengths.",
        "T");
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations", "dilation value along each spatial axis of the filter.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr("strides", "Stride along each spatial axis.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc2, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction(convShapeInference_opset1);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Conv,
    1,
    OpSchema().FillUsing(Conv_v1_FillSpec).TypeAndShapeInferenceFunction(convShapeInference_opset1));

static void convTransposeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return; // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
    if (kernel_shape.size() != n_input_dims) {
      return; // weight rank mismatch
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (size_t i = 0; i < kernel_shape.size(); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* const auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t total_pad = effective_kernel_shape[i] - strides[i];
        total_pad = std::max<int64_t>(total_pad, 0);
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) { // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto* final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) * group; // channels should be the second dim of second input
                                                                 // multiply group.

  int size_of_output = 0;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO(ONNX): throw exception?
          return; // output shape value cannot be smaller than the input shape
                  // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim = strides[i] * (input_shape.dim(i + 2).dim_value() - 1) + output_padding[i] +
            effective_kernel_shape[i] - pads[i] - pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator_opset1(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )DOC";
        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; has size (N x C x H x W)"
        ", where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn)",
        "T");
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (C x M/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "weight shape will be (C x M/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... x kn) is the dimension of the kernel. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T");
    schema.Input(2, "B", "Optional 1D bias to be added to the convolution, has size of M.", "T", OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the convolution. The "
        "output dimensions are functions of the kernel size, stride size, "
        "pad lengths and group count. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T");
    schema.TypeConstraint(
        "T", {types::Float16, types::Float, types::Double}, "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_shape",
        "The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified "
        "pads values are ignored. See doc for details for equations to generate pads",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_padding",
        "The zero-padding added to one side of the output."
        " This is also called adjs/adjustment in some frameworks.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations", "dilation value along each spatial axis of the filter.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr("strides", "Stride along each spatial axis.", AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr("auto_pad", auto_pad_doc2, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc2, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction(convTransposeShapeInference_opset1);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    1,
    OpSchema().FillUsing(ConvTranspose_v1_FillSpec).TypeAndShapeInferenceFunction(convTransposeShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(GlobalLpPool, 1, OpSchema().FillUsing(GlobalLpPool_v1_FillSpec));

static constexpr const char* BatchNormalization_ver1_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    1,
    OpSchema().FillUsing(BatchNormalization_v1_FillSpec).NumOutputs({1, 5}));

static constexpr const char* BatchNormalization_ver9_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    9,
    OpSchema()
        .FillUsing(BatchNormalization_v9_FillSpec)
        .NumOutputs({1, 5})
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static constexpr const char* BatchNormalization_ver14_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

Output case #1: Y, running_mean, running_var (training_mode=True)
Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B

where:

current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)

Notice that ReduceVar refers to the population variance, and it equals to
sum(sqrd(x_i - x_avg)) / N
where N is the population size (this formula does not use sample size N - 1).

```

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    14,
    OpSchema()
        .FillUsing(BatchNormalization_v14_FillSpec)
        .NumOutputs({1, 3})
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // Inputs 1 to 4 must be of rank 1.
          checkInputRank(ctx, 1, 1);
          checkInputRank(ctx, 2, 1);
          checkInputRank(ctx, 3, 1);
          checkInputRank(ctx, 4, 1);

          Dim num_channels;

          if (hasInputShape(ctx, 0)) {
            if (getInputShape(ctx, 0).dim_size() > 1)
              unifyInputDim(ctx, 0, 1, num_channels);
            else
              unifyDim(num_channels, 1);
          }

          unifyInputDim(ctx, 1, 0, num_channels);
          unifyInputDim(ctx, 2, 0, num_channels);
          unifyInputDim(ctx, 3, 0, num_channels);
          unifyInputDim(ctx, 4, 0, num_channels);

          if (ctx.getAttribute("training_mode") && static_cast<int>(ctx.getAttribute("training_mode")->i()) != 0) {
            if (ctx.getNumOutputs() != 3)
              fail_shape_inference("This number of op outputs should be 3 when Training_mode = True, but it is not.");
          } else {
            if (ctx.getNumOutputs() != 1)
              fail_shape_inference("This number of op outputs should be 1 when Training_mode = False, but it is not.");
          }

          if (ctx.getNumOutputs() > 1) {
            TensorShapeProto outputs_shape;
            *outputs_shape.add_dim() = num_channels; // channel

            propagateElemTypeFromInputToOutput(ctx, 3, 1);
            updateOutputShape(ctx, 1, outputs_shape);

            if (ctx.getNumOutputs() > 2) {
              propagateElemTypeFromInputToOutput(ctx, 4, 2);
              updateOutputShape(ctx, 2, outputs_shape);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(InstanceNormalization, 1, OpSchema().FillUsing(InstanceNormalization_v1_FillSpec));

static constexpr const char* Dropout_old_doc = R"DOC(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(Dropout, 1, OpSchema().FillUsing(Dropout_v1_FillSpec));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    6,
    OpSchema().FillUsing(Dropout_v6_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* const Dropout_ver7_doc = Dropout_old_doc;

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    7,
    OpSchema().FillUsing(Dropout_v7_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static constexpr const char* Dropout_ver10_doc = R"DOC(
Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    10,
    OpSchema().FillUsing(Dropout_v10_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
      if (ctx.getNumOutputs() == 2) {
        updateOutputElemType(ctx, 1, TensorProto::BOOL);
        if (hasNInputShapes(ctx, 1)) {
          propagateShapeFromInputToOutput(ctx, 0, 1);
        }
      }
    }));

static constexpr const char* BatchNorm_ver6_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    6,
    OpSchema()
        .FillUsing(BatchNormalization_v6_FillSpec)
        .NumOutputs({1, 5})
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    1,
    OpSchema().FillUsing(Flatten_v1_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    9,
    OpSchema().FillUsing(Flatten_v9_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0))
        return;
      auto& input_shape = getInputShape(ctx, 0);
      int rank = static_cast<int>(input_shape.dim_size());
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis > rank || axis < 0) {
        fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
      }
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));

static constexpr const char* BatchNormalization_ver7_doc = R"DOC(
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    there are multiple cases for the number of outputs, which we list below:

    Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
    Output case #2: Y (test mode)
        )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    7,
    OpSchema()
        .FillUsing(BatchNormalization_v7_FillSpec)
        .NumOutputs({1, 5})
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static constexpr const char* GroupNormalization_ver18_doc = R"DOC(
A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    18,
    OpSchema()
        .FillUsing(GroupNormalization_v18_FillSpec)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
              // GroupNormalization <epsilon, num_groups> (X, scale, bias) => (Y)
              auto tp = ctx.getInputType(0);
              if ((tp == nullptr) || (!tp->has_tensor_type()))
                return false;
              int64_t T = tp->tensor_type().elem_type();

              auto epsilon_attr = ctx.getAttribute("epsilon");
              float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;
              auto num_groups_attr = ctx.getAttribute("num_groups");
              if (num_groups_attr == nullptr)
                return false;
              int64_t num_groups = num_groups_attr->i();

              FunctionBuilder builder(functionProto);
              builder.Const1D("FloatEpsilon", epsilon)
                  .Add("Epsilon = Cast (FloatEpsilon)", "to", T)
                  .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
                  .Add("C = Shape <start = 1, end = 2> (X)")
                  .Const1D("NumGroups", num_groups)
                  .Add("GroupSize = Div (C, NumGroups)")
                  .Add("N = Shape <start = 0, end = 1> (X)") // batch size
                  .Add("InstanceShape = Shape <start = 2> (X)") // data instance shape

                  // NewShape = [N, num_groups, group_size, H, W, (...)]
                  .Add("NewShape = Concat <axis = 0> (N, NumGroups, GroupSize, InstanceShape)")
                  .Add("XReshaped = Reshape (X, NewShape)")

                  // Flatten into 3D tensor: [N, num_groups, group_size x H x W (x ...)]
                  .Add("Shape3D = Constant <value_ints = [0, 0, -1]> ()")
                  .Add("X3D = Reshape(XReshaped, Shape3D)")

                  // Calculate statistics
                  .Const1D("Axes2", static_cast<int64_t>(2))
                  .Add("Mean = ReduceMean (X3D, Axes2)")
                  .Add("Square = Mul (X3D, X3D)")
                  .Add("MeanOfSquare = ReduceMean (Square, Axes2)")
                  .Add("SquareOfMean = Mul (Mean, Mean)")
                  .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
                  .Add("VarPlusEpsilon = Add (Var, Epsilon)")
                  .Add("StdDev = Sqrt (VarPlusEpsilon)")
                  .Add("Deviation = Sub (X3D, Mean)")
                  .Add("Normalized = Div (Deviation, StdDev)")

                  // Reshape scale and bias for broadcasting
                  .Add("ScaleShape = Constant <value_ints = [1, -1, 1]> ()")
                  .Add("ScaleT = Cast (scale)", "to", T)
                  .Add("BiasT = Cast (bias)", "to", T)
                  .Add("ScaleReshaped = Reshape (ScaleT, ScaleShape)")
                  .Add("BiasReshaped = Reshape (BiasT, ScaleShape)")

                  // Calculate scaled and biased output
                  .Add("Scaled = Mul (ScaleReshaped, Normalized)")
                  .Add("Biased = Add (Scaled, BiasReshaped)")
                  .Add("Y = Reshape (Biased, XShape)");

              schema.BuildFunction(functionProto);
              return true;
            }));

static constexpr const char* Attention_ver23_doc = R"DOC(

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal attribute`, only one of which can be provided.
1) If `is_causal` is set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. The attention masking has the form of the upper left causal bias due to the alignment.
2) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
  softcap (if provided)     |
            |               |
 at_mask---Add              |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    23,
    OpSchema()
        .FillUsing(Attention_v23_FillSpec)
        .TypeAndShapeInferenceFunction(defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // ScaledDotProductAttention <scale, is_causal, q_num_heads, kv_numheads> (Q, K, V, attn_mask, past_key,
          // past_value) => (Y, present_key?, present_value?)
          int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
          int64_t float_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

          // Get input types
          auto t_qk = ctx.getInputType(0);
          if ((t_qk == nullptr) || (!t_qk->has_tensor_type()))
            return false;
          int64_t T1 = t_qk->tensor_type().elem_type();

          // Determine precision types for Softmax
          auto softmax_precision_attr = ctx.getAttribute("softmax_precision");
          int64_t softmax_precision = (softmax_precision_attr != nullptr) ? softmax_precision_attr->i() : T1;
          if ((softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
            return false; // Error

          // If shape is 3D, q_num_heads and kv_num_heads is provided,
          // for 4D cases, set num_heads to zero for reshape purposes
          auto q_num_heads_attr = ctx.getAttribute("q_num_heads");
          int64_t q_num_heads = (q_num_heads_attr != nullptr) ? q_num_heads_attr->i() : 0;
          auto kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
          int64_t kv_num_heads = (kv_num_heads_attr != nullptr) ? kv_num_heads_attr->i() : 0;

          // Determine if input is 3D (requires reshape and transpose) or 4D (direct reshape)
          bool is_3d_input = (q_num_heads > 0 && kv_num_heads > 0);

          FunctionBuilder builder(functionProto);
          builder
              .Add("BatchSize = Shape <start = 0, end = 1> (Q)") // batch size
              .Add("QSeqLen = Shape <start = -2, end = -1> (Q)") // q_sequence_length
              .Add("KVSeqLen = Shape <start = -2, end = -1> (K)"); // kv_sequence_length
          if (is_3d_input) {
            // For 3D inputs: First reshape to [batch_size, seq_length, num_heads, head_size]
            // then transpose to [batch_size, num_heads, seq_length, head_size]
            builder
                .Const1D("QNumHeadsAttr", q_num_heads) // q_num_heads from attrs
                .Const1D("KVNumHeadsAttr", kv_num_heads) // kv_num_heads from attrs
                .Const1D("NegOne", static_cast<int64_t>(-1)); // head_size, inferred from other dimensions

            builder.Add("QIntermediateShape = Concat <axis = 0> (BatchSize, QSeqLen, QNumHeadsAttr, NegOne)")
                .Add("KVIntermediateShape = Concat <axis = 0> (BatchSize, KVSeqLen, KVNumHeadsAttr, NegOne)")
                .Add("QIntermediate = Reshape (Q, QIntermediateShape)")
                .Add("KIntermediate = Reshape (K, KVIntermediateShape)")
                .Add("VIntermediate = Reshape (V, KVIntermediateShape)")
                // Then transpose to [batch_size, num_heads, seq_length, head_size]
                .Add("QReshaped = Transpose <perm = [0, 2, 1, 3]> (QIntermediate)")
                .Add("KReshaped = Transpose <perm = [0, 2, 1, 3]> (KIntermediate)")
                .Add("VReshaped = Transpose <perm = [0, 2, 1, 3]> (VIntermediate)");
          } else {
            // For 4D inputs: Already in desired shape [batch_size, num_heads, seq_length, head_size]
            builder.Add("QReshaped = Identity(Q)").Add("KReshaped = Identity(K)").Add("VReshaped = Identity(V)");
          }

          builder
              .Add("QNumHeads = Shape <start = 1, end = 2> (QReshaped)") // q_num_heads
              .Add("KVNumHeads = Shape <start = 1, end = 2> (KReshaped)"); // kv_num_heads

          // Calculate scaling factor if scale attribute not provided
          auto scale_attr = ctx.getAttribute("scale");
          float scale = (scale_attr != nullptr) ? scale_attr->f() : static_cast<float>(1);
          builder
              .Add("QKHeadSize = Shape <start = 3, end = 4> (QReshaped)") // head_size for Q and K
              .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", float_type)
              .Add("VHeadSize = Shape <start = 3, end = 4> (VReshaped)") // head_size for V
              .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
              .Const1D("One1D", static_cast<int64_t>(1))
              .Const1D("One1DF", static_cast<float>(1))
              .Const1D("Zero1D", static_cast<int64_t>(0))
              .Add("CalculatedScale = Div(One1DF, SqrtHeadSize)")
              .Const("ScaleF", ToTensor<float>(scale))
              .Add(scale_attr != nullptr ? "ScaleFactor = Identity(ScaleF)" : "ScaleFactor = Identity(CalculatedScale)")
              .Add("ScaleFactorSqrt = Sqrt(ScaleFactor)")
              .Add("ScaleFactorF = Cast (ScaleFactorSqrt)", "to", T1);

          // Update key and value caches for past and present states

          if (ctx.hasInput(4)) {
            builder.Add("PresentKey = Concat <axis = 2> (past_key, KReshaped)");
            builder.Add("PastKVSeqLen =  Shape <start = -2, end = -1> (past_key)");
          } else {
            builder.Add("PresentKey = Identity (KReshaped)");
            builder.Const1D("PastKVSeqLen", static_cast<int64_t>(0));
          }
          if (ctx.hasOutput(1)) {
            builder.Add("present_key = Identity (PresentKey)");
          }

          if (ctx.hasInput(5)) {
            builder.Add("PresentValue = Concat <axis = 2> (past_value, VReshaped)");
          } else {
            builder.Add("PresentValue = Identity (VReshaped)");
          }
          if (ctx.hasOutput(2)) {
            builder.Add("present_value = Identity (PresentValue)");
          }

          if (!defs::nn::utils::AttentionAppendFunctionCausalMask(ctx, builder, false))
            return false;
          builder.Add("AttnBiasT = Cast (AttnBiasCausalOrNot)", "to", T1);

          // Group Query Attention is applied if the following are satisfied
          // 1) q_num_heads != kv_num_heads
          // 2) q_num_heads % kv_num_heads == 0
          // 3) kv_num_heads == k_num_heads == v_num_heads
          builder.Add("NGQACond1 = Equal(QNumHeads, KVNumHeads)")
              .Add("GQACond1 = Not(NGQACond1)")
              .Add("DivNumHeads = Div(QNumHeads, KVNumHeads)")
              .Add("IDivNumHeads = Cast(DivNumHeads)", "to", int_type)
              .Add("RemainderNumHeads = Mod(QNumHeads, KVNumHeads)")
              .Add("GQACond2 = Equal(RemainderNumHeads, Zero1D)")
              .Add("GQACond = And(GQACond1, GQACond2)")
              .Add("InterleaveDim = Where(GQACond, IDivNumHeads, One1D)");

          // repeat kv (repeat_interleave)
          builder.Const1D("Two1D", static_cast<int64_t>(2))
              .Add("KUnsqueezed = Unsqueeze(PresentKey, Two1D)") // [B, Hk, 1, T, Dk]
              .Add("VUnsqueezed = Unsqueeze(PresentValue, Two1D)"); // [B, Hk, 1, T, Dv]

          // Build expand shape: [B, Hk, repeats, T, Dk]
          builder
              .Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, QKHeadSize)")
              .Add("KExpanded = Expand(KUnsqueezed, KExpandShape)");
          builder.Add("VExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, VHeadSize)")
              .Add("VExpanded = Expand(VUnsqueezed, VExpandShape)");

          // Reshape to [B, Hq, T, Dk] where Hq = Hk * repeats
          builder.Add("KAttentionShape = Concat <axis = 0> (BatchSize, QNumHeads, NewKVSeqLen, QKHeadSize)")
              .Add("VAttentionShape = Concat <axis = 0> (BatchSize, QNumHeads, NewKVSeqLen, VHeadSize)")
              .Add("KAttentionInput = Reshape(KExpanded, KAttentionShape)")
              .Add("VAttentionInput = Reshape(VExpanded, VAttentionShape)");

          // The following pattern is applied
          //      Q          K          V
          //      |          |          |
          //     Q*scale    K*scale     |
          //      |          |          |
          //      |       Transpose     |
          //      |          |          |
          //      ---MatMul---          |
          //            |               |
          //  softcap (if provided)     |
          //            |               |
          // at_mask---Add              |
          //            |               |
          //         Softmax            |
          //            |               |
          //            -----MatMul------
          //                    |
          //                    Y
          builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KAttentionInput)")
              .Add("QScaled = Mul(QReshaped, ScaleFactorF)")
              .Add("KScaled = Mul(KTranspose, ScaleFactorF)")
              .Add("QKAttnWeight = MatMul(QScaled, KScaled)")
              .Add("QKAttnCast = Cast (QKAttnWeight)", "to", T1);

          // Apply softcap before bias/mask addition.
          // Softcap must come before mask so that -inf mask values remain -inf
          // (zero probability in softmax). If applied after, -inf maps to
          // -softcap (finite), leaking probability to masked positions.
          auto softcap_attr = ctx.getAttribute("softcap");
          float softcap_val = (softcap_attr != nullptr) ? softcap_attr->f() : static_cast<float>(0);
          if (softcap_val != 0) {
            builder.Const1D("Softcap", softcap_val)
                .Add("SoftcapF = Cast (Softcap)", "to", T1)
                .Add("SoftcapDiv = Div(QKAttnCast, SoftcapF)")
                .Add("SoftcapTanh = Tanh(SoftcapDiv)")
                .Add("QKAttnSoftcapped = Mul(SoftcapTanh, SoftcapF)")
                .Add("QKAttnWeightSoftcap = Add(QKAttnSoftcapped, AttnBiasT)");
          } else {
            builder.Add("QKAttnWeightSoftcap = Add(QKAttnCast, AttnBiasT)");
          }
          builder.Add("SoftmaxCast = Cast (QKAttnWeightSoftcap)", "to", softmax_precision)
              .Add("AttnWeightSoftmax = Softmax (SoftmaxCast)")
              .Add("SoftmaxOut = Cast (AttnWeightSoftmax)", "to", T1);

          // QK MatMul output if required
          auto qk_matmul_output_mode_attr = ctx.getAttribute("qk_matmul_output_mode");
          int64_t qk_matmul_output_mode = (qk_matmul_output_mode_attr != nullptr) ? qk_matmul_output_mode_attr->i() : 0;
          if (ctx.hasOutput(3)) {
            if (qk_matmul_output_mode == 1) {
              // Mode 1: after softcap (before bias addition)
              if (softcap_val != 0) {
                builder.Add("qk_matmul_output = Identity(QKAttnSoftcapped)");
              } else {
                // No softcap applied, same as raw QK
                builder.Add("qk_matmul_output = Identity(QKAttnCast)");
              }
            } else if (qk_matmul_output_mode == 2) {
              // Mode 2: QK + softcap + bias (after softcap + bias addition)
              builder.Add("qk_matmul_output = Identity(QKAttnWeightSoftcap)");
            } else if (qk_matmul_output_mode == 3) {
              builder.Add("qk_matmul_output = Identity(AttnWeightSoftmax)");
            } else {
              builder.Add("qk_matmul_output = Identity(QKAttnWeight)");
            }
          }

          builder.Add("YPreReshape = MatMul(SoftmaxOut, VAttentionInput)");
          // Reshape Y to 3D if input is a 3D tensor
          if (is_3d_input) {
            builder.Add("YTranspose = Transpose <perm = [0, 2, 1, 3]> (YPreReshape)")
                .Add("YNewShape = Concat <axis = 0> (Zero1D, Zero1D, NegOne)")
                .Add("Y = Reshape(YTranspose, YNewShape)");
          } else {
            builder.Add("Y = Identity(YPreReshape)");
          }

          schema.BuildFunction(functionProto);
          return true;
        }));
} // namespace ONNX_NAMESPACE
