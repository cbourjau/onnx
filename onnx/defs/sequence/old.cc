// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/sequence/utils.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    SplitToSequence,
    11,
    OpSchema()
        .FillUsing(SplitToSequence_v11_FillSpec)
        .TypeAndShapeInferenceFunction(defs::sequence::utils::splitToSequenceShapeInference));

} // namespace ONNX_NAMESPACE
