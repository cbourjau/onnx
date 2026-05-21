// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
static constexpr const char* pads_doc =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
static constexpr const char* conv_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";
static constexpr const char* conv_transpose_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = input_shape[i] * strides[i]` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";

ONNX_API void convPoolShapeInference(
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
        if (i >= static_cast<int>(effective_kernel_shape.size())) {
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
    int64_t input_size = input_shape.dim(2 + i).dim_value();
    int64_t effective_input_size = input_size + pads[i] + pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    int64_t output_size =
        (effective_input_size - effective_kernel_shape[i] + (ceil_mode ? strides[i] - 1 : 0)) / strides[i] + 1;
    if (ceil_mode == 1 && (output_size - 1) * strides[i] >= (input_size + pads[i])) {
      // we need to match pytorch's behavior of "Sliding windows that would start in the right padded region are
      // ignored." (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#maxpool1d). this code follows the
      // same logic as PyTorch's C++ implementation:
      // https://github.com/pytorch/pytorch/blob/f1cdb39da3850c47d51ec6a5b1ae864c32b3accf/aten/src/ATen/native/Pool.h#L54C21-L54C21
      --output_size;
    }

    newdim->set_dim_value(output_size);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto* second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}
static std::function<void(InferenceContext&)> poolShapeInferenceFunc(bool use_dilation) {
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
    convPoolShapeInference(ctx, use_dilation, true, 0, 1);
  };
}

static void poolShapeInference_with_dilation(InferenceContext& ctx) {
  poolShapeInferenceFunc(true)(ctx);
}
ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    22,
    OpSchema().FillUsing(AveragePool_v22_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_with_dilation));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    22,
    OpSchema().FillUsing(MaxPool_v22_FillSpec).TypeAndShapeInferenceFunction(poolShapeInference_with_dilation));

static void maxUnpoolShapeInference(InferenceContext& ctx) {
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
    22,
    OpSchema().FillUsing(MaxUnpool_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      maxUnpoolShapeInference(ctx);
    }));

static void lpPoolShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference(ctx, true, true, 0, 1);
}
ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    22,
    OpSchema().FillUsing(LpPool_v22_FillSpec).TypeAndShapeInferenceFunction(lpPoolShapeInference));

// For ROI pool operations.
static void roiPoolTypeShapeInference(InferenceContext& ctx) {
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

static void roiPoolInference(InferenceContext& ctx) {
  roiPoolTypeShapeInference(ctx);
}
ONNX_OPERATOR_SET_SCHEMA(
    MaxRoiPool,
    22,
    OpSchema().FillUsing(MaxRoiPool_v22_FillSpec).TypeAndShapeInferenceFunction(roiPoolInference));

static void convShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  convPoolShapeInference(ctx, true, false, 0, 1);
}
ONNX_OPERATOR_SET_SCHEMA(
    Conv,
    22,
    OpSchema().FillUsing(Conv_v22_FillSpec).TypeAndShapeInferenceFunction(convShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    QLinearConv,
    10,
    OpSchema().FillUsing(QLinearConv_v10_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto x_type = ctx.getInputType(0);
      auto w_type = ctx.getInputType(3);
      if (nullptr == x_type || nullptr == w_type || x_type->value_case() != TypeProto::kTensorType ||
          w_type->value_case() != TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type in ", ctx.getDisplayName(), ".");
      }

      auto x_zero_point_type = ctx.getInputType(2);
      if (nullptr == x_zero_point_type ||
          x_zero_point_type->tensor_type().elem_type() != x_type->tensor_type().elem_type()) {
        fail_type_inference(
            "input and zero_point pair is expected to have be same type in ", ctx.getDisplayName(), ".");
      }

      auto w_zero_point_type = ctx.getInputType(5);
      if (nullptr == w_zero_point_type ||
          w_zero_point_type->tensor_type().elem_type() != w_type->tensor_type().elem_type()) {
        fail_type_inference("weight and zero_point pair is expected to have same type in ", ctx.getDisplayName(), ".");
      }

      propagateElemTypeFromInputToOutput(ctx, 7, 0);

      convPoolShapeInference(ctx, true, false, 0, 3);
    }));
ONNX_OPERATOR_SET_SCHEMA(
    ConvInteger,
    10,
    OpSchema().FillUsing(ConvInteger_v10_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto x_type = ctx.getInputType(0);
      auto w_type = ctx.getInputType(1);
      auto y_type = ctx.getOutputType(0);
      if (nullptr == x_type || nullptr == w_type || nullptr == y_type ||
          x_type->value_case() != TypeProto::kTensorType || w_type->value_case() != TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
      }

      // Right now we only support int32
      y_type->mutable_tensor_type()->set_elem_type(TensorProto::INT32);

      convPoolShapeInference(ctx, true, false, 0, 1);
    }));

ONNX_API void convTransposeShapeInference(InferenceContext& ctx) {
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

static void convTransposeInference(InferenceContext& ctx) {
  convTransposeShapeInference(ctx);
}
ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    22,
    OpSchema().FillUsing(ConvTranspose_v22_FillSpec).TypeAndShapeInferenceFunction(convTransposeInference));

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    22,
    OpSchema().FillUsing(DeformConv_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference(ctx, true, false, 0, 1);
    }));

// For GlobalPool operations.
ONNX_API void globalPoolTypeShapeInference(InferenceContext& ctx) {
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
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    22,
    OpSchema().FillUsing(GlobalAveragePool_v22_FillSpec).TypeAndShapeInferenceFunction(globalPoolTypeShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    GlobalMaxPool,
    22,
    OpSchema().FillUsing(GlobalMaxPool_v22_FillSpec).TypeAndShapeInferenceFunction(globalPoolTypeShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    22,
    OpSchema().FillUsing(GlobalLpPool_v22_FillSpec).TypeAndShapeInferenceFunction(globalPoolTypeShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    15,
    OpSchema()
        .FillUsing(BatchNormalization_v15_FillSpec)
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

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    22,
    OpSchema().FillUsing(InstanceNormalization_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    22,
    OpSchema().FillUsing(LpNormalization_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateShapeAndTypeFromFirstInput(ctx);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    22,
    OpSchema().FillUsing(Dropout_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    Shrink,
    9,
    OpSchema()
        .FillUsing(Shrink_v9_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Lambd = Constant <value_float: float = @lambd>()
            LambdCast = CastLike (Lambd, input)
            Bias = Constant <value_float: float = @bias>()
            BiasCast = CastLike (Bias, input)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, input)
            NegLmbda = Neg (LambdCast)
            InputLessThanNegLambda = Less (input, NegLmbda)
            InputAddBias = Add (input, BiasCast)
            InputSubBias = Sub (input, BiasCast)
            LambdaLessThanInput = Less (LambdCast, input)
            InputSubBiasOrZero = Where (LambdaLessThanInput, InputSubBias, ZeroCast)
            output = Where(InputLessThanNegLambda, InputAddBias, InputSubBiasOrZero)
          }
        )ONNX",
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    25,
    OpSchema().FillUsing(Flatten_v25_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      // TODO(ONNX): is the operation defined for input-rank < 2?
      updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
    }));
ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    13,
    OpSchema().FillUsing(LRN_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
ONNX_OPERATOR_SET_SCHEMA(
    TfIdfVectorizer,
    9,
    OpSchema().FillUsing(TfIdfVectorizer_v9_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
      output_elem_type->set_elem_type(TensorProto::FLOAT);

      if (hasInputShape(ctx, 0)) {
        std::vector<int64_t> ngram_indexes;
        getRepeatedAttribute(ctx, "ngram_indexes", ngram_indexes);
        if (ngram_indexes.empty() ||
            !std::all_of(ngram_indexes.cbegin(), ngram_indexes.cend(), [](int64_t i) { return i >= 0; })) {
          fail_shape_inference("ngram_indexes must be non-empty with no negative values");
        }

        auto greatest_hit = std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend());
        auto max_last_axis = *greatest_hit + 1;

        TensorShapeProto output_shape;
        auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        auto dim_size = input_shape.dim_size();
        if (dim_size == 1) {
          output_shape.add_dim()->set_dim_value(max_last_axis);
        } else if (dim_size == 2) {
          *output_shape.add_dim() = input_shape.dim(0);
          output_shape.add_dim()->set_dim_value(max_last_axis);
        } else {
          fail_shape_inference("Input tensor must have rank 1 or 2");
        }
        updateOutputShape(ctx, 0, output_shape);
      }
    }));
static const std::vector<int64_t> mvn_default_axes = {0, 2, 3};

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    13,
    OpSchema()
        .FillUsing(MeanVarianceNormalization_v13_FillSpec)
        .FunctionBody(R"ONNX(
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          X_RM = ReduceMean <axes : ints = @axes> (X)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean <axes : ints = @axes> (X_squared)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        )ONNX")
        .FunctionBody(
            R"ONNX(
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          axes = Constant <value_ints: ints = @axes>()
          X_RM = ReduceMean (X, axes)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean (X_squared, axes)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        )ONNX",
            18));

static void col2imShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // All inputs shapes are required
  if (!hasNInputShapes(ctx, 3)) {
    return;
  }

  // We assume image_shape has correct spatial dimensions for next validations
  // An alternative is get the the number of spatial dimensions as an input argument
  Dim n_input_dims;
  unifyInputDim(ctx, 1, 0, n_input_dims);

  unifyInputDim(ctx, 2, 0, n_input_dims);
  checkInputRank(ctx, 1, 1);
  checkInputRank(ctx, 2, 1);
  std::vector<int64_t> image_shape = {};
  const TensorProto* image_shape_data = ctx.getInputData(1);
  if (image_shape_data) {
    image_shape = ParseData<int64_t>(image_shape_data);
    unifyDim(n_input_dims, image_shape.size());
  }

  std::vector<int64_t> pads = {};
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() % 2) {
      fail_shape_inference("Attribute pads must have an even size");
    }
    unifyDim(n_input_dims, pads.size() / 2);
  }

  std::vector<int64_t> dilations = {};
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    unifyDim(n_input_dims, dilations.size());
  }

  std::vector<int64_t> strides = {};
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    unifyDim(n_input_dims, strides.size());
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() != 3) {
    fail_shape_inference("input must have rank 3.");
  }

  std::vector<int64_t> block_shape = {};
  const TensorProto* block_shape_data = ctx.getInputData(2);
  if (block_shape_data) {
    block_shape = ParseData<int64_t>(block_shape_data);
    unifyDim(n_input_dims, block_shape.size());
  }
  unifyInputDim(ctx, 2, 0, n_input_dims);

  int block_shape_size = 0;
  if (static_cast<int>(block_shape.size()) > 0) {
    block_shape_size = 1;
    for (const auto& dim : block_shape) {
      block_shape_size *= dim;
    }
  }
  // If we haven't inferred the number of image dimensions, we can't set inferred shape.
  if (!n_input_dims.has_dim_value()) {
    return;
  }

  // Final shape will be (N, C, dim_1, ..., dim_N)
  auto* final_image_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  // Dimensions N and C are always present
  Dim N, C;
  if (ctx.getInputType(0)->tensor_type().shape().dim(0).has_dim_value()) {
    N = input_shape.dim(0); // Otherwise, N is unknown.
  }
  *final_image_shape->add_dim() = N;

  if (block_shape_size > 0) {
    C = input_shape.dim(1) / block_shape_size; // Otherwise, C is unknown.
  }
  *final_image_shape->add_dim() = C;

  // Image dimensions are dynamic
  for (auto i = 0; i < n_input_dims.dim_value(); ++i) {
    Dim image_dim_i;
    if (!image_shape.empty()) {
      image_dim_i.set_dim_value(image_shape[i]); // Otherwise, spatial dimensions are unknown
    }
    *final_image_shape->add_dim() = image_dim_i;
  }
}
ONNX_OPERATOR_SET_SCHEMA(
    Col2Im,
    18,
    OpSchema().FillUsing(Col2Im_v18_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      col2imShapeInference(ctx);
    }));
static bool BuildContextDependentFunctionBodyLayerNormalization(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto,
    int sinceVersion) {
  ONNX_ASSERT(sinceVersion == 17 || sinceVersion == 18)
  // LayerNormalization <axis, epsilon, stash_type> (X, Scale, B) => (Y, Mean?, InvStdDev?)
  const auto* const tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  int64_t T = tp->tensor_type().elem_type();

  const auto* const type_attr = ctx.getAttribute("stash_type");
  int64_t U =
      (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16))
    return false; // Error

  const auto* const axis_attr = ctx.getAttribute("axis");
  int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
  const auto* const epsilon_attr = ctx.getAttribute("epsilon");
  float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

  auto mktensor = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
    auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
    tp.add_dims(1);
    return tp;
  };
  // The treatment of "axis" is different in "LayerNormalization" and in Reduction operations.
  // This complicates the function definition, requiring reshaping inputs/outputs.
  // Input X shape: [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]
  // This is treated as a 2D shape [d[0] * ... * d[axis-1], d[axis] * ... * d[rank-1]]
  // Normalization is applied to the second dimension.
  // Output Y has same shape as X
  // Outputs Mean and InvStdDev have shape: [d[0], ..., d[axis-1], 1, ..., 1]
  FunctionBuilder builder(functionProto);
  builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
      .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
      .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
      .Add("Rank = Size (XShape)") // rank of input tensor: scalar
      .Add("Zero1D = Constant()", "value", mktensor(0)) // [0] : 1D tensor
      .Add("Axis1D = Constant()", "value", mktensor(axis)) // [axis] : 1D tensor
      .Add("PrefixShape = Slice (XShape, Zero1D, Axis1D)") // [d[0], ..., d[axis-1]]
      .Add(
          axis >= 0 // number of axes that are reduced =
              ? "NumReducedAxes = Sub (Rank, Axis1D)" // [rank - axis]: 1D tensor
              : "NumReducedAxes = Neg (Axis1D)") // [-axis] : 1D tensor
      .Add(
          "SuffixShape = ConstantOfShape (NumReducedAxes)",
          "value",
          mktensor(1)) // [1, ..., 1] for reduced axes
      .Add("ReducedShape = Concat <axis = 0> (PrefixShape, SuffixShape)") // [d[0], ..., d[axis-1], 1, ..., 1]
      .Add("X2D = Flatten (X)", "axis", axis)
      .Add("XU = Cast (X2D)", "to", U);
  if (sinceVersion == 17) {
    builder.Add("Mean2D = ReduceMean <axes = [1]> (XU)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean <axes = [1]> (Square)");
  } else if (sinceVersion == 18) {
    builder.Add("Axes_1 = Constant()", "value", mktensor(1))
        .Add("Mean2D = ReduceMean (XU, Axes_1)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean (Square, Axes_1)");
  }
  builder.Add("SquareOfMean = Mul (Mean2D, Mean2D)")
      .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
      .Add("VarPlusEpsilon = Add (Var, Epsilon)")
      .Add("StdDev = Sqrt (VarPlusEpsilon)")
      .Add("Deviation = Sub (XU, Mean2D)")
      .Add("Normalized = Div (Deviation, StdDev)")
      .Add("NormalizedT = Cast (Normalized)", "to", T)
      .Add("Scale2D = Flatten <axis = 0> (Scale)")
      .Add("Scaled = Mul (NormalizedT, Scale2D)");
  if (ctx.hasInput(2)) {
    builder.Add("B2D = Flatten <axis=0> (B)");
    builder.Add("Biased = Add (Scaled, B2D)");
  } else {
    builder.Add("Biased = Identity (Scaled)");
  }
  builder.Add("Y = Reshape (Biased, XShape)");
  builder.Add("InvStdDev2D = Reciprocal (StdDev)");
  if (ctx.hasOutput(1))
    builder.Add("Mean = Reshape (Mean2D, ReducedShape)");
  if (ctx.hasOutput(2))
    builder.Add("InvStdDev = Reshape (InvStdDev2D, ReducedShape)");

  schema.BuildFunction(functionProto);
  return true;
}

static bool BuildContextDependentFunctionBodyLayerNormalizationVer17(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 17);
}

static bool BuildContextDependentFunctionBodyLayerNormalizationVer18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 18);
}

ONNX_OPERATOR_SET_SCHEMA(
    LayerNormalization,
    17,
    OpSchema()
        .FillUsing(LayerNormalization_v17_FillSpec)
        .AllowUncheckedAttributes()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer17, 17)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer18, 18)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          auto stash_type = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          auto stash_type_proto = ctx.getAttribute("stash_type");
          if (stash_type_proto) {
            stash_type = stash_type_proto->i();
          }
          if (ctx.getNumOutputs() > 1) {
            auto output_type = ctx.getOutputType(1);
            output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
          }
          if (ctx.getNumOutputs() > 2) {
            auto output_type = ctx.getOutputType(2);
            output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          int64_t input_ndim = input_shape.dim_size();
          int64_t axis = -1;
          auto axis_proto = ctx.getAttribute("axis");
          if (axis_proto) {
            axis = axis_proto->i();
          }
          if (axis < 0) {
            // Convert negative axis value to equivalent
            // positive value.
            axis += input_ndim;
          }
          if (axis < 0) {
            fail_shape_inference(
                "Unexpected axis value (",
                axis,
                ") rank of first input is ",
                input_ndim,
                " in ",
                ctx.getDisplayName(),
                ".");
          }
          if (ctx.getNumOutputs() > 1) {
            auto mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
            mean_shape->CopyFrom(input_shape);
            for (int d = static_cast<int>(axis); d < input_ndim; ++d)
              mean_shape->mutable_dim(d)->set_dim_value(1);
          }

          if (ctx.getNumOutputs() > 2) {
            auto inv_std_dev_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
            inv_std_dev_shape->CopyFrom(input_shape);
            for (int d = static_cast<int>(axis); d < input_ndim; ++d)
              inv_std_dev_shape->mutable_dim(d)->set_dim_value(1);
          }
        }));
ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    21,
    OpSchema()
        .FillUsing(GroupNormalization_v21_FillSpec)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
              // GroupNormalization <epsilon, num_groups> (X, scale, bias) => (Y)
              auto tp = ctx.getInputType(0);
              if ((tp == nullptr) || (!tp->has_tensor_type()))
                return false;
              int64_t in_type = tp->tensor_type().elem_type();

              auto epsilon_attr = ctx.getAttribute("epsilon");
              float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;
              auto num_groups_attr = ctx.getAttribute("num_groups");
              if (num_groups_attr == nullptr)
                return false;
              int64_t num_groups = num_groups_attr->i();

              auto stash_type_attr = ctx.getAttribute("stash_type");
              int64_t stash_type = (stash_type_attr != nullptr)
                  ? stash_type_attr->i()
                  : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              if ((stash_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
                  (stash_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
                  (stash_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
                  (stash_type != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
                return false; // Error

              FunctionBuilder builder(functionProto);
              builder.Const1D("FloatEpsilon", epsilon)
                  .Add("Epsilon = Cast (FloatEpsilon)", "to", stash_type)
                  .Add("XU = Cast (X)", "to", stash_type)
                  .Add("XShape = Shape (XU)") // shape of input tensor: 1D tensor
                  .Add("C = Shape <start = 1, end = 2> (X)")
                  .Const1D("NumGroups", num_groups)
                  .Add("GroupSize = Div (C, NumGroups)")
                  .Add("N = Shape <start = 0, end = 1> (X)") // batch size
                  .Add("InstanceShape = Shape <start = 2> (X)") // data instance shape

                  // NewShape = [N, num_groups, group_size, H, W, (...)]
                  .Add("NewShape = Concat <axis = 0> (N, NumGroups, GroupSize, InstanceShape)")
                  .Add("XReshaped = Reshape (XU, NewShape)")

                  // Flatten into 3D tensor: [N, num_groups, group_size x H x W (x ...)]
                  .Add("Shape3D = Constant <value_ints = [0, 0, -1]> ()")
                  .Add("X3D = Reshape (XReshaped, Shape3D)")

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
                  .Add("NormalizedU = Div (Deviation, StdDev)")

                  // Reshape to [N, C, H x W (x ...)] and cast to original type
                  .Add("NormalizedOriginalShape = Reshape (NormalizedU, XShape)")
                  .Add("NormalizedNC = Reshape (NormalizedOriginalShape, Shape3D)")
                  .Add("NormalizedT = Cast (NormalizedNC)", "to", in_type)

                  // Reshape scale and bias to [1, C, 1] for broadcasting
                  .Add("ScaleShape = Constant <value_ints = [1, -1, 1]> ()")
                  .Add("ScaleT = Cast (scale)", "to", in_type)
                  .Add("BiasT = Cast (bias)", "to", in_type)
                  .Add("ScaleReshaped = Reshape (ScaleT, ScaleShape)")
                  .Add("BiasReshaped = Reshape (BiasT, ScaleShape)")

                  // Calculate scaled and biased output
                  .Add("Scaled = Mul (ScaleReshaped, NormalizedT)")
                  .Add("Biased = Add (Scaled, BiasReshaped)")
                  .Add("Y = Reshape (Biased, XShape)");

              schema.BuildFunction(functionProto);
              return true;
            }));
ONNX_OPERATOR_SET_SCHEMA(
    RMSNormalization,
    23,
    OpSchema()
        .FillUsing(RMSNormalization_v23_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          int64_t input_ndim = input_shape.dim_size();
          int64_t axis = -1;
          auto axis_proto = ctx.getAttribute("axis");
          if (axis_proto) {
            axis = axis_proto->i();
          }
          if (axis < 0) {
            // Convert negative axis value to equivalent
            // positive value.
            axis += input_ndim;
          }
          if (axis < 0) {
            fail_shape_inference(
                "Unexpected axis value (",
                axis,
                ") rank of first input is ",
                input_ndim,
                " in ",
                ctx.getDisplayName(),
                ".");
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // RMSNormalization <axis, epsilon, stash_type> (X, Scale) => (Y)
          auto tp = ctx.getInputType(0);
          if ((tp == nullptr) || (!tp->has_tensor_type()))
            return false;
          int64_t T = tp->tensor_type().elem_type();

          auto type_attr = ctx.getAttribute("stash_type");
          int64_t U = (type_attr != nullptr) ? type_attr->i()
                                             : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
              (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
              (U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) && (U != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
            return false; // Error

          auto axis_attr = ctx.getAttribute("axis");
          int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
          auto epsilon_attr = ctx.getAttribute("epsilon");
          float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

          FunctionBuilder builder(functionProto);
          builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
              .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
              .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
              .Add("Rank = Size (XShape)") // rank of input tensor: scalar
              .Const("Axis", axis) // axis : scalar
              .Add(
                  axis >= 0 // number of axes that are reduced =
                      ? "PosAxis = Identity (Axis)" // axis: scalar
                      : "PosAxis = Add (Rank, Axis)") // rank + axis : scalar
              .Const("One", static_cast<int64_t>(1))
              .Add("ReduceAxes = Range(PosAxis, Rank, One)")
              .Add("XU = Cast (X)", "to", U);
          builder.Add("XSquared = Mul (XU, XU)")
              .Add("XSquaredMean = ReduceMean (XSquared, ReduceAxes)")
              .Add("MeanSquareEpsilon = Add (XSquaredMean, Epsilon)")
              .Add("RMS = Sqrt (MeanSquareEpsilon)")
              .Add("Normalized = Div (XU, RMS)")
              .Add("NormalizedT = Cast (Normalized)", "to", T);
          builder.Add("Y = Mul (NormalizedT, scale)");

          schema.BuildFunction(functionProto);
          return true;
        }));
ONNX_OPERATOR_SET_SCHEMA(
    RotaryEmbedding,
    23,
    OpSchema()
        .FillUsing(RotaryEmbedding_v23_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // we need at least one input to have a shape for this inference.
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          if ((input_shape.dim_size() < 3) || (input_shape.dim_size() > 4)) {
            fail_shape_inference("Input tensor must have at least 3 and at most 4 dimensions");
          }

          auto num_heads_attr = ctx.getAttribute("num_heads");
          if ((input_shape.dim_size() == 3) && (num_heads_attr == nullptr)) {
            fail_shape_inference("Input shape is 3D, num_heads attribute must be provided");
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // RotaryEmbedding <scale, interleaved, rotary_embedding_dim, num_heads> (X, position_ids, cos_cache,
          // sin_cache) => Y

          int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
          auto interleaved_attr = ctx.getAttribute("interleaved");
          int64_t interleaved = (interleaved_attr != nullptr) ? interleaved_attr->i() : 0;
          auto rotary_embedding_dim_attr = ctx.getAttribute("rotary_embedding_dim");
          int64_t rotary_embedding_dim = (rotary_embedding_dim_attr != nullptr) ? rotary_embedding_dim_attr->i() : 0;
          auto num_heads_attr = ctx.getAttribute("num_heads");
          int64_t num_heads = (num_heads_attr != nullptr) ? num_heads_attr->i() : 0;

          // Ensure that num_heads does not control reshaping of input tensor
          // when input tensor is 4D
          int64_t is_input_4d = 1;
          auto x_tp = ctx.getInputType(0);
          if ((x_tp == nullptr) || (!x_tp->has_tensor_type()))
            return false;
          if (!(x_tp->tensor_type().has_shape())) {
            return false;
          }
          if (x_tp->tensor_type().shape().dim_size() == 4) {
            is_input_4d = 1;
          } else if (x_tp->tensor_type().shape().dim_size() == 3) {
            is_input_4d = 0;
          } else {
            return false;
          }

          FunctionBuilder builder(functionProto);
          // Set input tensor to the correct shape if input shape is 3D
          // NewShape = [batch_size, sequence_length, num_heads, head_size]

          // Reshape tensor to 4D if input is 3D
          builder.Const1D("Zero1D", static_cast<int64_t>(0))
              .Const1D("NumHeads", num_heads) // num_heads
              .Const1D("NegOne", static_cast<int64_t>(-1)); // head_size, inferred from other dimensions

          if (is_input_4d == 0) {
            builder.Add("NewShape = Concat <axis = 0> (Zero1D, Zero1D, NumHeads, NegOne)")
                .Add("XIn = Reshape (X, NewShape)"); // new shape of input tensor: 4D tensor
          } else {
            builder.Add("XIn = Transpose <perm = [0, 2, 1, 3]> (X)");
          }

          // Rotary embedding dimension is the value along which the input is to be split
          // There are two cases for the rotary embedding dimension:
          // 1. Complete rotation: rotary embedding dimension defaults to head_size, rotary_embedding_dim = cos.shape[3]
          // * 2 or head_size
          // 2. Partial rotation: rotary embedding dimension is provided, rotary_embedding_dim = rotary_embedding_dim

          builder.Add("HeadSize = Shape <start = 3, end = 4> (XIn)");
          if (rotary_embedding_dim > 0) {
            builder.Const1D("RotaryEmbedDim", rotary_embedding_dim);
          } else {
            builder.Add("RotaryEmbedDim = Identity(HeadSize)");
          }
          builder.Const1D("Two1D", static_cast<int64_t>(2))
              .Add("NoRotateLength = Sub(HeadSize, RotaryEmbedDim)")
              .Add("RotateSplitLengths = Concat <axis = 0> (RotaryEmbedDim, NoRotateLength)");
          // shape of input to rotate = input[:,:,:,:rotary_embedding_dim]
          // shape of input not to rotate = input[:,:,:,rotary_embedding_dim:]
          builder.Add("XToRotate, XNoRotate = Split <axis = -1> (XIn, RotateSplitLengths)");

          // Gather the cos and sine matrices from the respective caches using position ids if provided.
          // Otherwise Gather op functions as an Identity op.
          // Unsqueeze applied to make cos and sin matrices have dimensions that are
          // valid for multiplication with input when is split. For cases where rotary_embedding_dim is provided,
          // slice the matrix values until that index only
          if (ctx.hasInput(3)) {
            builder
                .Add("CosCacheGather = Gather(cos_cache, position_ids)") // shape of cos matrix: [batch_size,
                                                                         // sequence_length, head_size / 2]
                .Add("SinCacheGather = Gather(sin_cache, position_ids)"); // shape of cos matrix: [batch_size,
                                                                          // sequence_length, head_size / 2]
          } else {
            builder
                .Add("CosCacheGather = Identity(cos_cache)") // shape of cos matrix: [batch_size, sequence_length,
                                                             // head_size / 2]
                .Add("SinCacheGather = Identity(sin_cache)"); // shape of cos matrix: [batch_size, sequence_length,
                                                              // head_size / 2]
          }

          builder.Add("RotaryEmbedDimHalf = Div(RotaryEmbedDim, Two1D)")
              .Add("RotaryEmbedDimHalfInt = Cast (RotaryEmbedDimHalf)", "to", int_type)
              .Add(
                  "CosCacheSliced = Slice(CosCacheGather, Zero1D, RotaryEmbedDimHalfInt, Two1D)") // shape of cos
                                                                                                  // matrix:
                                                                                                  // [batch_size,
                                                                                                  // sequence_length,
                                                                                                  // rotary_embedding_dim
                                                                                                  // / 2]
              .Add(
                  "SinCacheSliced = Slice(SinCacheGather, Zero1D, RotaryEmbedDimHalfInt, Two1D)") // shape of sin
                                                                                                  // matrix:
                                                                                                  // [batch_size,
                                                                                                  // sequence_length,
                                                                                                  // rotary_embedding_dim
                                                                                                  // / 2]
              .Add("CosCacheUnsqueezed = Unsqueeze(CosCacheSliced, Two1D)") // shape of cos matrix: [batch_size,
                                                                            // sequence_length, 1, rotary_embedding_dim
                                                                            // / 2]
              .Add("SinCacheUnsqueezed = Unsqueeze(SinCacheSliced, Two1D)"); // shape of sin matrix: [batch_size,
                                                                             // sequence_length, 1, rotary_embedding_dim
                                                                             // / 2]

          // Create slices of inputs to multiply with sin and cos matrices based on interleaved parameter
          // Choose the correct slices based on interleaved parameter
          // real = cos_x * x1 - sin_x * x2
          // imag = sin_x * x1 + cos_x * x2
          if (interleaved == 0) {
            // For non-interleaved (basic) rotation, slices are created as follows,
            builder.Add(
                "X1, X2 = Split <axis = -1, num_outputs = 2> (XToRotate)"); // shape of X1 =
                                                                            // input[:,:,:,:rotary_embedding_dim/2],
                                                                            // X2 =
                                                                            // input[:,:,:,rotary_embedding_dim/2:rotary_embedding_dim]
          } else {
            // For interleaved rotation, slices are created as follows,
            builder.Const1D("One1D", static_cast<int64_t>(1))
                .Const1D("AxesRotaryDim", static_cast<int64_t>(3))
                .Add("RotaryEmbedDimInclusive = Add(RotaryEmbedDim, One1D)")
                .Add(
                    "X1 = Slice(XToRotate, Zero1D, RotaryEmbedDim, AxesRotaryDim, Two1D)") // shape of X1 =
                                                                                           // input[:,:,:,0:rotary_embedding_dim:2]
                .Add(
                    "X2 = Slice(XToRotate, One1D, RotaryEmbedDimInclusive, AxesRotaryDim, Two1D)"); // shape of
                                                                                                    // X2 =
                                                                                                    // input[:,:,:,1:rotary_embedding_dim:2]
          }

          builder.Add("CosX1 = Mul(CosCacheUnsqueezed, X1)")
              .Add("SinX2 = Mul(SinCacheUnsqueezed, X2)")
              .Add("Real = Sub(CosX1, SinX2)")
              .Add("SinX1 = Mul(SinCacheUnsqueezed, X1)")
              .Add("CosX2 = Mul(CosCacheUnsqueezed, X2)")
              .Add("Imaginary = Add(SinX1, CosX2)");

          // Insert the real and imaginary values into the original input to be rotated based on interleaved parameter
          if (interleaved == 0) {
            builder.Add("XRotated = Concat <axis = -1> (Real, Imaginary)");
          } else {
            builder
                .Add("RealInterleave = Unsqueeze(Real, NegOne)") // shape of indices =
                                                                 // input[:,:,:,0:rotary_embedding_dim:2, 1]
                .Add("ImaginaryInterleave = Unsqueeze(Imaginary, NegOne)") // shape of indices =
                                                                           // input[:,:,:,1:rotary_embedding_dim+1:2, 1]
                .Add("XRotatedInterleavedConcat = Concat <axis = -1> (RealInterleave, ImaginaryInterleave)")
                .Add("XRotatedShape = Shape(XToRotate)")
                .Add("XRotated = Reshape(XRotatedInterleavedConcat, XRotatedShape)");
          }

          // Combine rotated parts with non-rotated parts
          builder.Add("XConcat = Concat <axis = -1> (XRotated, XNoRotate)");

          if (is_input_4d == 0) {
            builder.Add("YTransposed = Identity(XConcat)");
          } else {
            builder.Add("YTransposed = Transpose <perm = [0, 2, 1, 3]> (XConcat)");
          }
          // Reshape back to 3D shape if input is a 3D tensor
          builder.Add("XShape = Shape(X)").Add("Y = Reshape(YTransposed, XShape)");

          schema.BuildFunction(functionProto);
          return true;
        }));
ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    24,
    OpSchema()
        .FillUsing(Attention_v24_FillSpec)
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
              .Const1D("NegOne1D", static_cast<int64_t>(-1))
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

          if (!defs::nn::utils::AttentionAppendFunctionCausalMask(ctx, builder, true))
            return false;

          // Add padding mask if kv_nonpad_seqlen is provided
          if (ctx.hasInput(6)) {
            builder
                .Add("KVSeqLenExpanded = Unsqueeze(nonpad_kv_seqlen, One1D)") // [batch_size, 1]
                .Add("KVSeqLen0D = Squeeze(KVSeqLen)")
                .Const("Zero0D", static_cast<int64_t>(0))
                .Const("One0D", static_cast<int64_t>(1))
                .Add("Range = Range(Zero0D, KVSeqLen0D, One0D)") // [KVSeqLen,]
                .Add("PaddingMaskBool = Less(Range, KVSeqLenExpanded)") // [batch_size, KVSeqLen]
                .Add("PaddingMaskFloat = Where(PaddingMaskBool, ScalarZero, FloatNegInf)") // [batch_size, KVSeqLen]
                .Add("PaddingMask3D = Unsqueeze(PaddingMaskFloat, One1D)") // [batch_size, 1, KVSeqLen]
                .Add("PaddingMask4D = Unsqueeze(PaddingMask3D, One1D)") // [batch_size, 1, 1, KVSeqLen]
                .Add("AttnBiasCausalPad = Add(AttnBiasCausalOrNot, PaddingMask4D)");
          } else {
            builder.Add("AttnBiasCausalPad = Identity(AttnBiasCausalOrNot)");
          }
          builder.Add("AttnBiasT = Cast (AttnBiasCausalPad)", "to", T1);

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
