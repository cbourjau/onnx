// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
ONNX_API void RNNShapeInference(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  auto layout_value = getAttribute(ctx, "layout", 0);

  if (hasInputShape(ctx, 0)) {
    const auto& first_input_shape = getInputShape(ctx, 0);
    if (first_input_shape.dim_size() != 3) {
      fail_shape_inference("First input tensor must have rank 3");
    }
    seq_length = first_input_shape.dim((layout_value == 0) ? 0 : 1);
    batch_size = first_input_shape.dim((layout_value == 0) ? 1 : 0);
  }

  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs > 0) {
    // Y
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    if (layout_value == 0) {
      auto dims = {seq_length, num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 0, dims);
    } else {
      auto dims = {batch_size, seq_length, num_directions, hidden_size};
      updateOutputShape(ctx, 0, dims);
    }
  }

  if (num_outputs > 1) {
    // Y_h
    propagateElemTypeFromInputToOutput(ctx, 0, 1);

    if (layout_value == 0) {
      auto dims = {num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 1, dims);
    } else {
      auto dims = {batch_size, num_directions, hidden_size};
      updateOutputShape(ctx, 1, dims);
    }
  }

  if (num_outputs > 2) {
    // Y_c : only in the case of LSTM
    propagateElemTypeFromInputToOutput(ctx, 0, 2);

    if (layout_value == 0) {
      auto dims = {num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 2, dims);
    } else {
      auto dims = {batch_size, num_directions, hidden_size};
      updateOutputShape(ctx, 2, dims);
    }
  }
}

static std::function<void(OpSchema&)> RNNDocGenerator(const char* /*name*/) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("forward"));
    schema.Attr(
        "layout",
        "The shape format of inputs X, initial_h and outputs Y, Y_h. "
        "If 0, the following shapes are expected: "
        "X.shape = [seq_length, batch_size, input_size], "
        "Y.shape = [seq_length, num_directions, batch_size, hidden_size], "
        "initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size]. "
        "If 1, the following shapes are expected: "
        "X.shape = [batch_size, seq_length, input_size], "
        "Y.shape = [batch_size, seq_length, num_directions, hidden_size], "
        "initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators."
        "For example with LeakyRelu, the default alpha is 0.01.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "clip",
        "Cell clip threshold. Clipping bounds the elements of a tensor "
        "in the range of [-threshold, +threshold] and is applied to the input "
        "of activations. No clip if not specified.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        4,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        5,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.TypeConstraint("T1", {types::Int32}, "Constrain seq_lens to integer tensor.");
    schema.TypeAndShapeInferenceFunction(RNNShapeInference);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    RNN,
    22,
    OpSchema().FillUsing(RNN_v22_FillSpec).TypeAndShapeInferenceFunction(RNNShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    GRU,
    22,
    OpSchema().FillUsing(GRU_v22_FillSpec).TypeAndShapeInferenceFunction(RNNShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    LSTM,
    22,
    OpSchema().FillUsing(LSTM_v22_FillSpec).TypeAndShapeInferenceFunction(RNNShapeInference));
} // namespace ONNX_NAMESPACE
