// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <string>
#include <vector>

#include "onnx/defs/controlflow/utils.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;

static std::vector<std::string> control_flow_types_ir13() {
  auto t = OpSchema::all_tensor_types_ir13();
  auto s = OpSchema::all_tensor_sequence_types_ir13();
  auto o = OpSchema::all_optional_types_ir13();
  t.insert(t.end(), s.begin(), s.end());
  t.insert(t.end(), o.begin(), o.end());
  return t;
}

ONNX_OPERATOR_SET_SCHEMA(
    If,
    25,
    OpSchema().FillUsing(If_v25_FillSpec).TypeAndShapeInferenceFunction(IfInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Loop,
    25,
    OpSchema().FillUsing(Loop_v25_FillSpec).TypeAndShapeInferenceFunction(LoopInferenceFunction));

ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    25,
    OpSchema()
        .FillUsing(Scan_v25_FillSpec)
        .TypeAndShapeInferenceFunction(ScanInferenceFunction)); // Shares same shape inference as opset 11

} // namespace ONNX_NAMESPACE
