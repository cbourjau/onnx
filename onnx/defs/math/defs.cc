// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/op_specs_generated.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static void MathOpDataPropagator(DataPropagationContext& ctx, const std::string& op_type) {
  const auto* const input_0 = ctx.getInputData(0);
  const auto* const input_1 = ctx.getInputData(1);
  if (input_0 == nullptr || input_1 == nullptr) {
    return;
  }
  int size_0 = input_0->dim_size();
  int size_1 = input_1->dim_size();
  // Fails to broadcast if the ranks are different and no any rank is 1
  if (size_0 != size_1 && size_0 != 1 && size_1 != 1) {
    fail_shape_inference("Invalid rank for ", op_type, " broadcasting: (", size_0, ") vs (", size_1, ").");
  }
  TensorShapeProto tsp;
  int size_out = size_0 == 1 ? size_1 : size_0;
  for (int i = 0; i < size_out; ++i) {
    const auto& input_dim_0 = input_0->dim(size_0 == 1 ? 0 : i);
    const auto& input_dim_1 = input_1->dim(size_1 == 1 ? 0 : i);
    if (input_dim_0.has_dim_value() && input_dim_1.has_dim_value()) {
      tsp.mutable_dim()->Add()->set_dim_value(
          defs::math::utils::MathOpTwoIntegers(op_type, input_dim_0.dim_value(), input_dim_1.dim_value()));
    } else {
      // Cannot compute the value; simply add an empty dim without value and param
      tsp.mutable_dim()->Add();
    }
  }
  ctx.addOutputData(0, std::move(tsp));
}

static void binaryBroadcastShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 2))
    bidirectionalBroadcastShapeInference(
        ctx.getInputType(0)->tensor_type().shape(),
        ctx.getInputType(1)->tensor_type().shape(),
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}
ONNX_OPERATOR_SET_SCHEMA(
    Add,
    14,
    OpSchema()
        .FillUsing(Add_v14_FillSpec)
        .TypeAndShapeInferenceFunction(binaryBroadcastShapeInference)
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Add"); }));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    14,
    OpSchema()
        .FillUsing(Sub_v14_FillSpec)
        .TypeAndShapeInferenceFunction(binaryBroadcastShapeInference)
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Sub"); }));
ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    13,
    OpSchema().FillUsing(Mod_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    14,
    OpSchema()
        .FillUsing(Mul_v14_FillSpec)
        .TypeAndShapeInferenceFunction(binaryBroadcastShapeInference)
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Mul"); }));

ONNX_OPERATOR_SET_SCHEMA(
    Div,
    14,
    OpSchema().FillUsing(Div_v14_FillSpec).TypeAndShapeInferenceFunction(binaryBroadcastShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    13,
    OpSchema().FillUsing(Neg_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    13,
    OpSchema().FillUsing(Abs_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    13,
    OpSchema().FillUsing(Reciprocal_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    13,
    OpSchema().FillUsing(Floor_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    13,
    OpSchema().FillUsing(Ceil_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    13,
    OpSchema().FillUsing(Sqrt_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    14,
    OpSchema()
        .FillUsing(Relu_v14_FillSpec)
        .FunctionBody(
            R"ONNX(
          {
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            Y = Max (X, ZeroCast)
          }
        )ONNX",
            18)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    16,
    OpSchema()
        .FillUsing(LeakyRelu_v16_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike(Zero, X)
            XLessThanZero = Less(X, ZeroCast)
            AlphaMulX = Mul (AlphaCast, X)
            Y = Where (XLessThanZero, AlphaMulX, X)
          }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    22,
    OpSchema()
        .FillUsing(ThresholdedRelu_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            AlphaLessThanX = Less(AlphaCast, X)
            Y = Where(AlphaLessThanX, X, ZeroCast)
          }
        )ONNX",
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    22,
    OpSchema()
        .FillUsing(Selu_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Gamma = Constant <value_float: float = @gamma>()
            GammaCast = CastLike (Gamma, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            ExpX = Exp (X)
            AlphaMulExpX = Mul(AlphaCast, ExpX)
            AlphaMulExpXSubAlpha = Sub (AlphaMulExpX, AlphaCast)
            Neg = Mul (GammaCast, AlphaMulExpXSubAlpha)
            Pos = Mul (GammaCast, X)
            XLessThanZero = Less (X, ZeroCast)
            Y = Where(XLessThanZero, Neg, Pos)
          }
        )ONNX",
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    22,
    OpSchema()
        .FillUsing(Elu_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            One = Constant <value = float {1.0}>()
            OneCast = CastLike (One, X)
            XLessThanZero = Less (X, ZeroCast)
            ExpX = Exp (X)
            ExpXSubOne = Sub (ExpX, OneCast)
            AlphaMulExpXSubOne = Mul (AlphaCast, ExpXSubOne)
            Y = Where(XLessThanZero, AlphaMulExpXSubOne, X)
          }
        )ONNX",
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Mish,
    22,
    OpSchema()
        .FillUsing(Mish_v22_FillSpec)
        .FunctionBody(R"ONNX(
          {
            Softplus_X = Softplus (X)
            TanHSoftplusX = Tanh (Softplus_X)
            Y = Mul (X, TanHSoftplusX)
           }
        )ONNX")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
static float celu_default_alpha = 1.0;

static bool BuildContextDependentFunctionBodyCelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  float alpha = ctx.getAttribute("alpha") != nullptr ? ctx.getAttribute("alpha")->f() : celu_default_alpha;
  FunctionBuilder builder(functionProto);
  builder.Const("alpha", std::vector<float>{alpha}).Add(R"(
            X_alpha = Div (X, alpha)
            Elu_Result = Elu <alpha = 1.0>(X_alpha)
            Y = Mul (alpha, Elu_Result)
        )");
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    12,
    OpSchema()
        .FillUsing(Celu_v12_FillSpec)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyCelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
static constexpr const char* gelu_default_approx = "none";

static bool BuildContextDependentFunctionBodyGelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* const approx_attr_proto = ctx.getAttribute("approximate");
  std::string approximate =
      approx_attr_proto != nullptr && approx_attr_proto->has_s() ? approx_attr_proto->s() : gelu_default_approx;
  FunctionBuilder builder(functionProto);

  if (approximate == "tanh") {
    builder.Add(R"(
              Half = Constant <value = float {0.5}>()
              HalfCast = CastLike (Half, X)
              One = Constant <value = float {1.0}>()
              OneCast = CastLike (One, X)
              TwoOverPi = Constant <value = float {0.63661977236}>()
              TwoOverPiCast = CastLike (TwoOverPi, X)
              C0 = Constant <value = float {0.044715}>()
              C0Cast = CastLike (C0, X)
              SqrtTwoOverPi = Sqrt (TwoOverPiCast)
              Three = Constant <value = float {3.0}>()
              ThreeCast = CastLike (Three, X)
              XCubed = Pow (X, ThreeCast)
              XCubedC0 = Mul (C0Cast, XCubed)
              XC0XCubed = Sum (X, XCubedC0)
              TanhInput = Mul (SqrtTwoOverPi, XC0XCubed)
              ErfApprox = Tanh (TanhInput)
              PhiApprox = Sum (OneCast, ErfApprox)
              MultX = Mul (HalfCast, X)
              Y = Mul (MultX, PhiApprox)
              )");
  } else {
    builder.Add(R"(
              Half = Constant <value = float {0.5}>()
              HalfCast = CastLike (Half, X)
              One = Constant <value = float {1.0}>()
              OneCast = CastLike (One, X)
              Two = Constant <value = float {2.0}>()
              TwoCast = CastLike (Two, X)
              SqrtTwo = Sqrt (TwoCast)
              XSqrt = Div (X, SqrtTwo)
              ErfXSqrt = Erf(XSqrt)
              Phi = Sum (OneCast, ErfXSqrt)
              MultX = Mul (HalfCast, X)
              Y = Mul (MultX, Phi)
              )");
  }
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Gelu,
    20,
    OpSchema()
        .FillUsing(Gelu_v20_FillSpec)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyGelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
ONNX_OPERATOR_SET_SCHEMA(
    Swish,
    24,
    OpSchema()
        .FillUsing(Swish_v24_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
            {
                Alpha = Constant <value_float: float = @alpha>()
                AlphaCast = CastLike (Alpha, X)
                AlphaMulX = Mul (AlphaCast, X)
                SigmoidAlphaMulX = Sigmoid(AlphaMulX)
                Y = Mul (X, SigmoidAlphaMulX)
            }
            )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    13,
    OpSchema().FillUsing(Exp_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    13,
    OpSchema().FillUsing(Log_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    13,
    OpSchema().FillUsing(Tanh_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    15,
    OpSchema().FillUsing(Pow_v15_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    16,
    OpSchema()
        .FillUsing(PRelu_v16_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
        {
          Zero = Constant <value = float {0.0}>()
          ZeroCast = CastLike(Zero, X)
          XLessThanZero = Less (X, ZeroCast)
          SlopeMulX = Mul (slope, X)
          Y = Where(XLessThanZero, SlopeMulX, X)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    13,
    OpSchema().FillUsing(Sigmoid_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    22,
    OpSchema()
        .FillUsing(HardSigmoid_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Alpha = Constant <value_float: float = @alpha>()
            AlphaCast = CastLike (Alpha, X)
            Beta = Constant <value_float: float = @beta>()
            BetaCast = CastLike (Beta, X)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, X)
            One = Constant <value = float {1.0}>()
            OneCast = CastLike (One, X)
            AlphaMulX = Mul (X, AlphaCast)
            AlphaMulXAddBeta = Add (AlphaMulX, BetaCast)
            MinOneOrAlphaMulXAddBeta = Min (AlphaMulXAddBeta, OneCast)
            Y = Max(MinOneOrAlphaMulXAddBeta, ZeroCast)
          }
        )ONNX",
            18));

ONNX_OPERATOR_SET_SCHEMA(
    HardSwish,
    22,
    OpSchema()
        .FillUsing(HardSwish_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(R"ONNX(
          {
            HS_X = HardSigmoid<alpha = 0.16666667163372, beta = 0.5>(X)
            Y = Mul (X, HS_X)
          }
        )ONNX"));

static void elementwiseMultiOpShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  int num_inputs = static_cast<int>(ctx.getNumInputs());
  std::vector<const TensorShapeProto*> shapes;
  shapes.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    const auto* const input_type = ctx.getInputType(i);
    if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
      return;
    }
    shapes.push_back(&input_type->tensor_type().shape());
  }

  multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
ONNX_OPERATOR_SET_SCHEMA(
    Max,
    13,
    OpSchema().FillUsing(Max_v13_FillSpec).TypeAndShapeInferenceFunction(elementwiseMultiOpShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    13,
    OpSchema().FillUsing(Min_v13_FillSpec).TypeAndShapeInferenceFunction(elementwiseMultiOpShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    13,
    OpSchema().FillUsing(Sum_v13_FillSpec).TypeAndShapeInferenceFunction(elementwiseMultiOpShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    13,
    OpSchema().FillUsing(Mean_v13_FillSpec).TypeAndShapeInferenceFunction(elementwiseMultiOpShapeInference));
static bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  bool has_min = ctx.hasInput(1);
  bool has_max = ctx.hasInput(2);

  FunctionBuilder builder(functionProto);
  if (!has_min && !has_max) {
    builder.Add("output = Identity (input)");
  } else if (has_min && !has_max) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("output = Where (input_less_than_min, min, input)");
  } else if (!has_min && has_max) {
    builder.Add("input_large_than_max = Less (max, input)");
    builder.Add("output = Where (input_large_than_max, max, input)");
  } else {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("tmp = Where (input_less_than_min, min, input)");
    builder.Add("output_large_than_max = Less (max, tmp)");
    builder.Add("output = Where (output_large_than_max, max, tmp)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    13,
    OpSchema()
        .FillUsing(Clip_v13_FillSpec)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyClip)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void softmaxShapeInference(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference starts
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  // Validate the value of 'axis'
  const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  int r = input_shape.dim_size();
  if (r == 0) {
    fail_shape_inference("Input rank must be >= 1 for softmax family op.");
  }
  int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
  if (axis < -r || axis >= r) {
    fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
  }

  // Shape inference
  propagateShapeFromInputToOutput(ctx, 0, 0);
}
ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    13,
    OpSchema()
        .FillUsing(Softmax_v13_FillSpec)
        .TypeAndShapeInferenceFunction(softmaxShapeInference)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis)
                  .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                  .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

              schema.BuildFunction(functionProto);
              return true;
            })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            18));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    13,
    OpSchema()
        .FillUsing(LogSoftmax_v13_FillSpec)
        .TypeAndShapeInferenceFunction(softmaxShapeInference)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis)
                  .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                  .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            13)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    13,
    OpSchema().FillUsing(Hardmax_v13_FillSpec).TypeAndShapeInferenceFunction(softmaxShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    22,
    OpSchema()
        .FillUsing(Softsign_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            One = Constant <value = float {1.0}>()
            OneCast = CastLike (One, input)
            AbsInput = Abs(input)
            OneAddAbsInput = Add (OneCast, AbsInput)
            output = Div(input, OneAddAbsInput)
          }
        )ONNX",
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    22,
    OpSchema()
        .FillUsing(Softplus_v22_FillSpec)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
            {
              exp_x = Exp (X)
              one = Constant <value = float {1.0}>()
              one_cast = CastLike (one, X)
              exp_x_add_one = Add (exp_x, one_cast)
              Y = Log (exp_x_add_one)
            }
            )ONNX",
            18));
ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    13,
    OpSchema().FillUsing(Gemm_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (first_input_shape.dim_size() != 2) {
          fail_shape_inference("First input does not have rank 2");
        }
        if (second_input_shape.dim_size() != 2) {
          fail_shape_inference("Second input does not have rank 2");
        }
        updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    13,
    OpSchema().FillUsing(MatMul_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      defs::math::utils::MatMulShapeInference(ctx, 0, 1);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    24,
    OpSchema().FillUsing(TopK_v24_FillSpec).TypeAndShapeInferenceFunction(defs::math::utils::topKShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    22,
    OpSchema().FillUsing(Sin_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    22,
    OpSchema().FillUsing(Cos_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    22,
    OpSchema().FillUsing(Tan_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    22,
    OpSchema().FillUsing(Asin_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    22,
    OpSchema().FillUsing(Acos_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    22,
    OpSchema().FillUsing(Atan_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    13,
    OpSchema().FillUsing(Expand_v13_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      // For shape inference, we need both input shape
      if (hasNInputShapes(ctx, 2)) {
        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        bool found = false;
        TensorShapeProto second_shape = getShapeInput(ctx, 1, found);
        if (found) {
          bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    22,
    OpSchema().FillUsing(Sinh_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    22,
    OpSchema().FillUsing(Cosh_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    22,
    OpSchema().FillUsing(Asinh_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    22,
    OpSchema().FillUsing(Acosh_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    22,
    OpSchema().FillUsing(Atanh_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    13,
    OpSchema().FillUsing(Sign_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    13,
    OpSchema().FillUsing(Erf_v13_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    21,
    OpSchema()
        .FillUsing(QLinearMatMul_v21_FillSpec)
        .TypeAndShapeInferenceFunction(defs::math::utils::QLinearMatMulShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    MatMulInteger,
    10,
    OpSchema()
        .FillUsing(MatMulInteger_v10_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto a_type = ctx.getInputType(0);
          auto b_type = ctx.getInputType(1);
          auto y_type = ctx.getOutputType(0);
          if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
              a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
          }

          // Right now we only support int32
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

          defs::math::utils::MatMulShapeInference(ctx, 0, 1);
        }));
ONNX_OPERATOR_SET_SCHEMA(
    CumProd,
    26,
    OpSchema()
        .FillUsing(CumProd_v26_FillSpec)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    CumSum,
    14,
    OpSchema()
        .FillUsing(CumSum_v14_FillSpec)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    22,
    OpSchema().FillUsing(Round_v22_FillSpec).TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    22,
    OpSchema().FillUsing(Det_v22_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        const int rank = static_cast<int>(input_shape.dim_size());

        if (rank < 2) {
          fail_shape_inference("Input rank must be >= 2.");
        }

        const auto mat_w = input_shape.dim(rank - 1);
        const auto mat_h = input_shape.dim(rank - 2);
        if (mat_w.has_dim_value() && mat_h.has_dim_value() && (mat_w.dim_value() != mat_h.dim_value())) {
          fail_shape_inference(
              "The inner-most 2 dimensions must have the same size (mat_w:",
              mat_w.dim_value(),
              " != mat_h:",
              mat_h.dim_value(),
              ").");
        }

        for (int i = 0; i < rank - 2; ++i) {
          auto dim = output_shape->add_dim();
          *dim = input_shape.dim(i);
        }
      }
    }));

static bool BuildContextDependentFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  const auto* const reduction_attr_proto = ctx.getAttribute("reduction");
  std::string reduction_attr =
      reduction_attr_proto != nullptr && reduction_attr_proto->has_s() ? reduction_attr_proto->s() : "mean";

  FunctionBuilder builder(functionProto);
  builder.Const1D("const_zero", static_cast<int64_t>(0))
      .Const1D("const_one", static_cast<int64_t>(1))
      .Const1D("axes", static_cast<int64_t>(1))
      .Add("expanded_target = Unsqueeze (target, axes)");

  if (ctx.getAttribute("ignore_index") == nullptr) {
    builder.Add(R"(
      input_gather_element = GatherElements <axis = 1> (input, expanded_target)
      loss_NCdd = Neg (input_gather_element)
      loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)
    )");

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        builder.Add("loss = Squeeze (loss_N1dd, axes)");
      } else {
        builder.Add("loss_Ndd = Squeeze (loss_N1dd, axes)");
        if (reduction_attr == "mean") {
          builder.Add("loss = ReduceMean <keepdims = 0> (loss_Ndd)");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    } else {
      builder.Add("weight_gather = Gather (weight, target)");
      builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
      if (reduction_attr == "none") {
        builder.Add("loss = Mul (loss_unweighted, weight_gather)");
      } else {
        builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
        if (reduction_attr == "mean") {
          builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
          )");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    }
  } else {
    builder.Const1D("const_ignore_index", ctx.getAttribute("ignore_index")->i());
    builder.Add(R"(
      const_zero_target_typed = Sub (expanded_target, expanded_target)
      expanded_target_int64 = Cast <to = 7> (expanded_target)
      mask = Equal (expanded_target_int64, const_ignore_index)
      transform_targets = Where (mask, const_zero_target_typed, expanded_target)
    )");
    builder.Add("input_gather_element = GatherElements <axis = 1> (input, transform_targets)");
    builder.Const1D("const_zero_float", 0.0f);
    if (!float_input) {
      builder.Add("const_zero_casted = Cast (const_zero_float)", "to", static_cast<int64_t>(input_type))
          .Add("input_gather_element_transform = Where (mask, const_zero_casted, input_gather_element)");
    } else {
      builder.Add("input_gather_element_transform = Where (mask, const_zero_float, input_gather_element)");
    }
    builder.Add("loss_NCdd = Neg (input_gather_element_transform)");
    builder.Add("loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)");

    if (!ctx.hasInput(2)) {
      builder.Add("squeeze_mask = Squeeze (mask, axes)");
      builder.Const1D("const_one_float", 1.0f);
      if (!float_input) {
        builder.Add("const_one_casted = Cast (const_one_float)", "to", static_cast<int64_t>(input_type))
            .Add("weight_gather = Where (squeeze_mask, const_zero_casted, const_one_casted)");
      } else {
        builder.Add("weight_gather = Where (squeeze_mask, const_zero_float, const_one_float)");
      }

    } else {
      builder.Add("weight_gather_temp = Gather (weight, transform_targets)");
      builder.Add(
          float_input ? "weight_gather_temp_1 = Where (mask, const_zero_float, weight_gather_temp)"
                      : "weight_gather_temp_1 = Where (mask, const_zero_casted, weight_gather_temp)");
      builder.Add("weight_gather = Squeeze (weight_gather_temp_1, axes)");
    }

    builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
    if (reduction_attr == "none") {
      builder.Add("loss = Mul (loss_unweighted, weight_gather)");
    } else {
      builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
      if (reduction_attr == "mean") {
        builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
        )");
      } else {
        builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
      }
    }
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    22,
    OpSchema()
        .FillUsing(NegativeLogLikelihoodLoss_v22_FillSpec)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBody)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasNInputShapes(ctx, 2)) {
            const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const TensorShapeProto& target_shape = ctx.getInputType(1)->tensor_type().shape();

            const int input_rank = static_cast<int>(input_shape.dim_size());
            const int target_rank = static_cast<int>(target_shape.dim_size());

            if (input_rank < 2) {
              fail_shape_inference("Input rank must be >= 2. input_rank=", input_rank);
            }
            if (target_rank != input_rank - 1) {
              fail_shape_inference(
                  "Target rank must be 1 less than the input rank. input_rank=",
                  input_rank,
                  ", target_rank=",
                  target_rank);
            }

            // match input dimensions (N, C, d1, ..., dk) with target
            // dimensions of (C, d1, ..., dk)
            for (int dim = 0; dim < target_rank; dim++) {
              const auto input_dim = dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
              const auto target_dim = target_shape.dim(dim);
              if (input_dim.has_dim_value() && target_dim.has_dim_value() &&
                  input_dim.dim_value() != target_dim.dim_value())
                fail_shape_inference(
                    "Input and target dimension value mismatch. input_dim_value=",
                    input_dim.dim_value(),
                    " target_dim_value=",
                    target_dim.dim_value());
            }

            if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
              const TensorShapeProto& weight_shape = ctx.getInputType(2)->tensor_type().shape();
              const auto weight_rank = weight_shape.dim_size();
              if (weight_rank != 1) {
                fail_shape_inference("Weight rank must be 1. weight_rank=", weight_rank);
              }
            }

            TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

            if (getAttribute(ctx, "reduction", "mean") == "none") {
              // output tensor is of shape (N, d1, d2, ..., dk) if
              // reduction attribute is "none".
              for (int i = 0; i < input_rank - 1; i++) {
                auto dim = output_shape->add_dim();
                if (i == 0)
                  *dim = input_shape.dim(i);
                else
                  *dim = input_shape.dim(i + 1);
              }
            }
            // otherwise output is a scalar.
          }
        }));

static void einsumShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, std::string const& equation) {
  // Only accept letters for indices
  auto is_letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };

  const size_t num_inputs = ctx.getNumInputs();
  if (num_inputs < 1 || !hasNInputShapes(ctx, num_inputs)) {
    return;
  }
  ONNX_NAMESPACE::TensorShapeProto output_shape;
  std::string left_equation;

  auto mid_index = equation.find("->");
  if (mid_index != std::string::npos) {
    // Separate right and left hand sides of the equation
    left_equation = equation.substr(0, mid_index);
  } else {
    // No right hand side
    left_equation = equation;
  }

  std::string term;
  size_t num_operands = 0;
  size_t num_ellipsis = 0;
  size_t num_ellipsis_indices = 0;

  // Parse the left-hand side
  std::stringstream str(left_equation);
  std::map<char, size_t> label_maps;
  std::unordered_set<char> repeated_labels;
  ONNX_NAMESPACE::TensorShapeProto dims_value, ellipsis_dims_value;
  size_t num_labels = 0;
  bool ellipsis_flag = true;

  while (!str.eof()) {
    std::getline(str, term, ',');
    auto ellipsis_index = term.find("...");
    if (num_inputs <= num_operands) {
      fail_shape_inference("Number of input tensors does not match the operands in the equation.");
    }
    const auto& shape = ctx.getInputType(num_operands)->tensor_type().shape();
    size_t rank = shape.dim_size();
    size_t ellipsis_dims = 0;

    size_t term_size = 0; // number of legal indices for the current term
    size_t num_illegal_char = 0; // number of illegal char before the current 'index' in the current term

    for (char index : term) {
      if (is_letter(index)) {
        term_size += 1;
      }
    }

    // Validate that term_size is compatible with rank before accessing dimensions
    if (ellipsis_index != std::string::npos) {
      // For ellipsis case, rank must be at least term_size
      if (rank < term_size) {
        fail_shape_inference(
            "Ellipsis represents incompatible dimensions for input ",
            num_operands,
            ". Rank ",
            rank,
            " is less than term size ",
            term_size,
            ".");
      }
    } else {
      // For non-ellipsis case, rank must equal term_size
      if (rank != term_size) {
        fail_shape_inference(
            "Rank of input ", num_operands, " (", rank, ") does not match the equation indices (", term_size, ").");
      }
    }

    for (size_t index = 0; index < term.size(); ++index) {
      if (index == ellipsis_index) {
        // find ellipsis and record the dims represented by ellipsis
        ellipsis_dims = rank - term_size;
        if (ellipsis_flag) {
          ellipsis_flag = false;
          for (size_t i = 0; i < ellipsis_dims; i++) {
            *ellipsis_dims_value.add_dim() = shape.dim(index + i - num_illegal_char);
          }
        } else {
          for (size_t i = 0; i < ellipsis_dims; i++) {
            const auto shape_dim = shape.dim(index + i - num_illegal_char);
            auto* const current_dim = ellipsis_dims_value.mutable_dim(i);
            if (shape_dim.has_dim_value() && current_dim->has_dim_value() &&
                shape_dim.dim_value() > current_dim->dim_value() && current_dim->dim_value() == 1) {
              current_dim->set_dim_value(shape_dim.dim_value());
            }
          }
        }
        index += 2; // skip the rest of dots
        num_illegal_char += 3;
        continue;

      } else if (!is_letter(term[index])) {
        num_illegal_char += 1;
        continue;
      }

      const auto inserted = label_maps.emplace(term[index], num_labels).second;
      if (inserted) {
        *dims_value.add_dim() = shape.dim(index + ellipsis_dims - num_illegal_char);
        ++num_labels;
      } else {
        repeated_labels.insert(term[index]);
      }
    }

    if (ellipsis_index != std::string::npos) {
      // If there is an ellipsis, the number of dimensions it represents
      // must be total dim - letter dimensions
      if (num_ellipsis == 0) {
        num_ellipsis_indices = rank - term_size;
      } else { // ellipsis has been seen before. Check that if dimensions
               // are compatible
        if (num_ellipsis_indices != rank - term_size) {
          fail_shape_inference("Ellipsis represents incompatible dimensions.");
        }
      }
      num_ellipsis++;
    }
    num_operands++;
  }

  if (num_inputs != num_operands) {
    fail_shape_inference("Number of input tensors does not match the operands in the equation.");
  }

  // Parse the provided right-hand side
  if (mid_index != std::string::npos) {
    std::string right_equation = equation.substr(mid_index + 2);
    auto right_ellipsis_index = right_equation.find("...");

    for (size_t index = 0; index < right_equation.size(); ++index) {
      // If there's an ellipsis, add its corresponding dimensions
      if (index == right_ellipsis_index) {
        for (size_t i = 0; i < num_ellipsis_indices; i++) {
          *output_shape.add_dim() = ellipsis_dims_value.dim(i);
        }
        index += 2; // skip the rest of dots
        continue;
      }

      if (is_letter(right_equation[index])) {
        *output_shape.add_dim() = dims_value.dim(label_maps[right_equation[index]]);
      }
    }
  } else { // Infer the dimension for right-hand side
    // If there's an ellipsis, add its corresponding dimensions
    for (size_t i = 0; i < num_ellipsis_indices; i++) {
      *output_shape.add_dim() = ellipsis_dims_value.dim(i);
    }
    // If no explicit output was given, generate an implicit output by ordering all the
    // labels in alphabetic order (by ASCII value consistent with numpy, so Z < a).
    // Exclude any labels that occurred more than once, as these cancel out.
    for (const auto& [label, dim_idx] : label_maps) {
      if (repeated_labels.count(label) == 0) {
        *output_shape.add_dim() = dims_value.dim(dim_idx);
      }
    }
  }

  updateOutputShape(ctx, 0, output_shape);
}
ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    12,
    OpSchema().FillUsing(Einsum_v12_FillSpec).TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      std::string equation = getAttribute(ctx, "equation", "");
      if (equation.empty()) {
        return;
      }

      equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                     equation.end()); // Remove space char
      einsumShapeInference(ctx, equation);
    }));
static bool BuildContextDependentFunctionBodySCE(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  // Using stable implementation of LogSoftmax
  builder //
      .Const("Shape3D", std::vector<int64_t>({0, 0, -1})) //
      .Add(R"(
        X_NCD = Reshape (scores, Shape3D)
        X_NDC = Transpose <perm = [0, 2, 1]> (X_NCD)
        X_LogSM = LogSoftmax <axis = 2> (X_NDC)
        X_LogSM_NCD = Transpose <perm = [0, 2, 1]> (X_LogSM)
        X_shape = Shape (scores)
        X_Log = Reshape (X_LogSM_NCD, X_shape)
      )");

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    builder.Add("log_prob = Identity (X_Log)");
  }

  builder.Add(
      ctx.hasInput(2)
          ? "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels, weights)"
          : "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels)");

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    13,
    OpSchema()
        .FillUsing(SoftmaxCrossEntropyLoss_v13_FillSpec)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySCE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string reduction = getAttribute(ctx, "reduction", "mean");
          if (reduction == "none") {
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          } else {
            updateOutputShape(ctx, 0, TensorShapeProto());
          }

          if (ctx.getNumOutputs() == 2) {
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }));
ONNX_OPERATOR_SET_SCHEMA(
    DFT,
    20,
    OpSchema().FillUsing(DFT_v20_FillSpec).TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
      bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

      const size_t input_arg_index = 0;
      const size_t dft_length_arg_index = 1;
      const size_t axis_arg_index = 2;
      const size_t output_index = 0;

      propagateElemTypeFromInputToOutput(ctx, input_arg_index, output_index);
      if (!hasInputShape(ctx, input_arg_index)) {
        // If no shape is available for the input, skip shape inference...
        return;
      }

      auto& input_shape = getInputShape(ctx, input_arg_index);

      // The last dimension is the real and imaginary parts of the value.
      const int64_t rank = input_shape.dim_size();
      if (rank < 2) {
        fail_shape_inference("input tensor must have rank >= 2, including the complex dimension.");
      }

      // check the inputs are correct types for one-sided DFT
      if (is_onesided) {
        auto last_dim = input_shape.dim(rank - 1);
        if (inverse) {
          // Check last dimension is 2 (complex input required)
          if (last_dim.has_dim_value() && last_dim.dim_value() != 2) {
            fail_shape_inference("inverse one-sided DFT requires complex input (last dimension must be 2)");
          }
        } else {
          // Check last dimension is 1 (real input required)
          if (last_dim.has_dim_value() && last_dim.dim_value() != 1) {
            fail_shape_inference("one-sided DFT requires real input (last dimension must be 1)");
          }
        }
      }

      // In general the output shape will match the input shape exactly
      // So initialize the output shape with the input shape
      TensorShapeProto result_shape_proto = input_shape;

      // Handle when axis is an input but is not statically known
      if (ctx.hasInput(axis_arg_index) && ctx.getInputData(axis_arg_index) == nullptr) {
        // Axis is an input but is not statically known
        if (is_onesided || ctx.hasInput(dft_length_arg_index)) {
          // We don't know which axis is the signal dimension, so we cannot infer shape
          // when onesided is enabled or when dft_length_arg_index is provided
          TensorShapeProto new_shape_proto{};
          for (int i = 0; i < rank; ++i) {
            new_shape_proto.add_dim();
          }
          // Set last dimension based on operation type
          ONNX_ASSERTM(
              rank == static_cast<int64_t>(new_shape_proto.dim_size()),
              "rank should be equal to new_shape_proto.dim_size()")
          if (inverse && is_onesided) {
            // IRFFT: output is real-valued
            new_shape_proto.mutable_dim(rank - 1)->set_dim_value(1);
          } else {
            // Complex output
            new_shape_proto.mutable_dim(rank - 1)->set_dim_value(2);
          }
          updateOutputShape(ctx, output_index, new_shape_proto);
          return;
        } else {
          // Set last dimension based on operation type
          int dim_size = result_shape_proto.dim_size();
          if (inverse && is_onesided) {
            // IRFFT: output is real-valued
            result_shape_proto.mutable_dim(dim_size - 1)->set_dim_value(1);
          } else {
            // Complex output
            result_shape_proto.mutable_dim(dim_size - 1)->set_dim_value(2);
          }
          updateOutputShape(ctx, output_index, result_shape_proto);
          return;
        }
      }

      // Get the axis where the DFT will be performed
      int64_t axis;
      if (!ctx.hasInput(axis_arg_index)) {
        // axis is not an input. We use -2 by default
        axis = -2;
      } else {
        const TensorProto* axis_tensor = ctx.getInputData(axis_arg_index);
        ONNX_ASSERTM(axis_tensor != nullptr, "axis should not be nullptr at this point")
        // TODO(justinchuby): Create invariance checking functions to ensure shapes and sizes
        // to abstract the following logic out.
        if (axis_tensor->dims_size() != 0) {
          fail_shape_inference("axis input must be a scalar.");
        }
        axis = defs::math::utils::GetScalarValueFromTensor<int64_t>(axis_tensor);
      }

      // NOLINTNEXTLINE(readability-simplify-boolean-expr)
      if (!(-rank <= axis && axis != -1 && axis < rank - 1)) {
        fail_shape_inference(
            "axis attribute value ",
            axis,
            " is invalid for a tensor of rank ",
            rank,
            ". Valid values are '-rank <= axis && axis != -1 && axis < rank - 1'");
      }

      auto axis_idx = (axis >= 0 ? axis : axis + rank);

      // If dft_length is specified, then we should honor the shape.
      // Set the output dimension to match the dft_length on the axis.
      if (ctx.hasInput(dft_length_arg_index)) {
        // dft_length is provided
        const TensorProto* dft_length = ctx.getInputData(dft_length_arg_index);
        if (dft_length == nullptr) {
          // If we cannot read the dft_length, we cannot infer shape on the signal axis
          result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
        } else {
          if (dft_length->dims_size() != 0) {
            fail_shape_inference("dft_length input must be a scalar.");
          }
          auto dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

          // For RFFT, output size on signal axis is floor(dft_length/2) + 1
          if (is_onesided && !inverse) {
            // RFFT: one-sided output
            auto half_signal_size = (dft_length_value >> 1) + 1;
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
          } else {
            // Standard FFT/IFFT and IRFFT: full length
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(dft_length_value);
          }
        }
      } else if (is_onesided) {
        auto axis_dimension = result_shape_proto.dim(axis_idx);
        if (axis_dimension.has_dim_value()) {
          auto axis_dimension_value = axis_dimension.dim_value();
          if (inverse) {
            // IRFFT without explicit dft_length: cannot reliably infer full signal length
            // Default to even length: N = 2 * (input_size - 1)
            auto full_signal_size = 2 * (axis_dimension_value - 1);
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(full_signal_size);
          } else {
            // RFFT without explicit dft_length: infer one-sided output size from input
            auto half_signal_size = (axis_dimension_value >> 1) + 1;
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
          }
        } else {
          result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
          result_shape_proto.mutable_dim(axis_idx)->clear_dim_param();
        }
      }

      // Set the last dimension based on whether output is real or complex
      if (is_onesided && inverse) {
        // IRFFT: complex input -> real output (last dim = 1)
        result_shape_proto.mutable_dim(static_cast<int>(rank - 1))->set_dim_value(1);
      } else {
        // All other cases: complex output (last dim = 2)
        result_shape_proto.mutable_dim(static_cast<int>(rank - 1))->set_dim_value(2);
      }

      updateOutputShape(ctx, output_index, result_shape_proto);
    }));

static void cosineSumWindowShapeInference(InferenceContext& ctx) {
  // Update the output data type to the output_datatype
  auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
  updateOutputElemType(ctx, 0, output_datatype);

  if (!hasInputShape(ctx, 0)) {
    // If no shape is available for the input, skip shape inference.
    return;
  }

  const auto* const size = ctx.getInputData(0);
  if (size == nullptr) {
    // Size is not available, so return early
    return;
  }

  if (size->dims_size() != 0) {
    fail_shape_inference("size input must be a scalar.");
  }

  auto size_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(size);
  if (size_value <= 0) {
    fail_shape_inference("size input must be greater than 0.");
  }

  ONNX_NAMESPACE::TensorShapeProto result_shape;
  result_shape.add_dim()->set_dim_value(size_value);
  updateOutputShape(ctx, 0, result_shape);
}
ONNX_OPERATOR_SET_SCHEMA(
    HannWindow,
    17,
    OpSchema()
        .FillUsing(HannWindow_v17_FillSpec)
        .TypeAndShapeInferenceFunction(cosineSumWindowShapeInference)
        .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.5}>()
          A1 = Constant <value = float {0.5}>()
          A2 = Constant <value = float {0.0}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Periodic_Size_FP = Cast <to = 1> (size)
          Symmetric_Size_FP = Sub(Periodic_Size_FP, One)
          IsPeriodic = Constant <value_int : int = @periodic>()
          IsPeriodic_FP = Cast <to = 1> (IsPeriodic)
          IsSymmetric_FP = Sub(One, IsPeriodic_FP)
          Periodic_Component = Mul(Periodic_Size_FP, IsPeriodic_FP)
          Symmetric_Component = Mul(Symmetric_Size_FP, IsSymmetric_FP)
          Size_FP = Add(Periodic_Component, Symmetric_Component)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Periodic_Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Sub (A0, A1_Component)
          Temp1 = Add (Temp0, A2_Component)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    HammingWindow,
    17,
    OpSchema()
        .FillUsing(HammingWindow_v17_FillSpec)
        .TypeAndShapeInferenceFunction(cosineSumWindowShapeInference)
        .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.54347826087}>()
          A1 = Constant <value = float {0.45652173913}>()
          A2 = Constant <value = float {0.0}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Periodic_Size_FP = Cast <to = 1> (size)
          Symmetric_Size_FP = Sub(Periodic_Size_FP, One)
          IsPeriodic = Constant <value_int : int = @periodic>()
          IsPeriodic_FP = Cast <to = 1> (IsPeriodic)
          IsSymmetric_FP = Sub(One, IsPeriodic_FP)
          Periodic_Component = Mul(Periodic_Size_FP, IsPeriodic_FP)
          Symmetric_Component = Mul(Symmetric_Size_FP, IsSymmetric_FP)
          Size_FP = Add(Periodic_Component, Symmetric_Component)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Periodic_Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Sub (A0, A1_Component)
          Temp1 = Add (Temp0, A2_Component)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX"));

ONNX_OPERATOR_SET_SCHEMA(
    BlackmanWindow,
    17,
    OpSchema()
        .FillUsing(BlackmanWindow_v17_FillSpec)
        .TypeAndShapeInferenceFunction(cosineSumWindowShapeInference)
        .FunctionBody(R"ONNX(
        {
          A0 = Constant <value = float {0.42}>()
          A1 = Constant <value = float {0.5}>()
          A2 = Constant <value = float {0.08}>()
          Zero = Constant <value = float {0.0}>()
          One = Constant <value = float {1.0}>()
          Two = Constant <value = float {2.0}>()
          Tau = Constant <value = float {6.2831853}>()
          Periodic_Size_FP = Cast <to = 1> (size)
          Symmetric_Size_FP = Sub(Periodic_Size_FP, One)
          IsPeriodic = Constant <value_int : int = @periodic>()
          IsPeriodic_FP = Cast <to = 1> (IsPeriodic)
          IsSymmetric_FP = Sub(One, IsPeriodic_FP)
          Periodic_Component = Mul(Periodic_Size_FP, IsPeriodic_FP)
          Symmetric_Component = Mul(Symmetric_Size_FP, IsSymmetric_FP)
          Size_FP = Add(Periodic_Component, Symmetric_Component)
          AngularIncrement = Div (Tau, Size_FP)
          Range = Range (Zero, Periodic_Size_FP, One)
          RangeAngular = Mul (Range, AngularIncrement)
          TwoRangeAngular = Mul (RangeAngular, Two)
          CosTwoRangeAngular = Cos (TwoRangeAngular)
          A2_Component = Mul (A2, CosTwoRangeAngular)
          CosRangeAngular = Cos (RangeAngular)
          A1_Component = Mul (A1, CosRangeAngular)
          Temp0 = Sub (A0, A1_Component)
          Temp1 = Add (Temp0, A2_Component)
          output = Cast <to : int = @output_datatype> (Temp1)
        }
        )ONNX"));
ONNX_OPERATOR_SET_SCHEMA(
    MelWeightMatrix,
    17,
    OpSchema()
        .FillUsing(MelWeightMatrix_v17_FillSpec)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
          updateOutputElemType(ctx, 0, output_datatype);

          if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 1)) {
            return;
          }

          const auto num_mel_bins = ctx.getInputData(0);
          const auto dft_length = ctx.getInputData(1);
          if (nullptr == num_mel_bins || nullptr == dft_length) {
            return;
          }

          int64_t num_mel_bins_value = -1;
          int64_t dft_length_value = -1;
          if (num_mel_bins->dims_size() != 0) {
            fail_shape_inference("num_mel_bins input must be scalar.");
          }
          num_mel_bins_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(num_mel_bins);

          if (dft_length->dims_size() != 0) {
            fail_shape_inference("dft_length input must be scalar.");
          }
          dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

          if (num_mel_bins_value > 0 && dft_length_value > 0) {
            ONNX_NAMESPACE::TensorShapeProto result_shape;
            result_shape.add_dim()->set_dim_value(static_cast<int64_t>((dft_length_value >> 1) + 1));
            result_shape.add_dim()->set_dim_value(num_mel_bins_value);
            updateOutputShape(ctx, 0, result_shape);
          }
        }));
ONNX_OPERATOR_SET_SCHEMA(
    STFT,
    17,
    OpSchema().FillUsing(STFT_v17_FillSpec).TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Get signal size
      // The signal size is needed to perform inference because the size of the signal
      // is needed to compute the number of DFTs in the output.
      //
      // 1) Check if shape exists, return if not
      // 2) Get the shape
      // 3) Check if signal dim value exists, return if not
      if (!hasInputShape(ctx, 0)) {
        return;
      }

      auto& input_shape = getInputShape(ctx, 0);
      if (input_shape.dim_size() < 2) {
        fail_shape_inference("First input should have at least 2 dimensions in ", ctx.getDisplayName(), ".");
      }
      auto signal_dim = input_shape.dim(1);
      if (!signal_dim.has_dim_value()) {
        return;
      }
      auto signal_size = signal_dim.dim_value();

      // The frame step is a required input.
      // Its value is needed to compute the number output nDFTs, so return early is missing.
      const auto frame_step = ctx.getInputData(1);
      if (nullptr == frame_step) {
        return;
      }
      auto frame_step_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_step);

      // Determine the size of the DFT based on the 2 optional inputs window and frame_length.
      // One must be set.
      int64_t dft_size = -1;
      const TensorProto* frame_length = nullptr;
      if (ctx.hasInput(3)) {
        frame_length = ctx.getInputData(3);
        if (frame_length == nullptr) {
          // If we cannot read the frame_length, we cannot infer shape
          // return...
          return;
        }
      }

      const TensorShapeProto* window_shape = nullptr;
      if (ctx.getNumInputs() >= 3) {
        window_shape = getOptionalInputShape(ctx, 2);
      } else {
        window_shape = nullptr;
      }

      if (window_shape == nullptr && frame_length == nullptr) {
        // STFT expects to have at least one of these inputs set: [window, frame_length],
        // but they may not be available at shape inference time
        return;
      } else if (window_shape != nullptr && frame_length != nullptr) {
        if (frame_length->dims_size() != 0) {
          fail_shape_inference("frame_length input must be scalar.");
        }
        auto frame_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_length);

        // Ensure that the window length and the dft_length match.
        if (window_shape->dim_size() != 1) {
          fail_shape_inference("window input must have rank = 1.");
        }
        if (window_shape->dim(0).has_dim_value()) {
          auto window_length = window_shape->dim(0).dim_value();
          if (window_length != frame_length_value) {
            fail_type_inference(
                "If STFT has both a window input and frame_length specified, the dimension of the window must match the frame_length specified!");
          }
        }

        dft_size = frame_length_value;
      } else if (window_shape != nullptr) {
        // Ensure that the window length and the dft_length match.
        if (window_shape->dim_size() != 1) {
          fail_shape_inference("window input must have rank = 1.");
        }
        if (window_shape->dim(0).has_dim_value()) {
          dft_size = window_shape->dim(0).dim_value();
        } else {
          // Cannot determine the window size, and there is no frame_length,
          // So shape inference cannot proceed.
          return;
        }
      } else if (frame_length != nullptr) {
        if (frame_length->dims_size() != 0) {
          fail_shape_inference("frame_length input must be scalar.");
        }
        dft_size = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_length);
      }

      bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
      int64_t dft_unique_bins = is_onesided ? ((dft_size >> 1) + 1) : dft_size;

      auto n_dfts = static_cast<int64_t>((signal_size - dft_size) / static_cast<float>(frame_step_value)) + 1;

      // The output has the following shape: [batch_size][frames][dft_unique_bins][2]
      ONNX_NAMESPACE::TensorShapeProto result_shape_proto;
      auto batch_dim = result_shape_proto.add_dim();

      if (input_shape.dim(0).has_dim_value()) {
        batch_dim->set_dim_value(input_shape.dim(0).dim_value()); // batch size
      }

      result_shape_proto.add_dim()->set_dim_value(n_dfts);
      result_shape_proto.add_dim()->set_dim_value(dft_unique_bins);
      result_shape_proto.add_dim()->set_dim_value(2);
      updateOutputShape(ctx, 0, result_shape_proto);
    }));
} // namespace ONNX_NAMESPACE
