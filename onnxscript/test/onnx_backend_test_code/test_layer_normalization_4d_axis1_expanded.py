
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset17


@script()
def bck_test_layer_normalization_4d_axis1_expanded(X: FLOAT[2, 3, 4, 5], W: FLOAT[3, 4, 5], B: FLOAT[3, 4, 5]) -> (FLOAT[2, 3, 4, 5], FLOAT[2, 1, 1, 1], FLOAT[2, 1, 1, 1]):

    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_FloatEpsilon = opset17.Constant(
        value=make_tensor("value", 1, dims=[], vals=[9.999999747378752e-06]))
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Epsilon = opset17.Cast(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_FloatEpsilon, to=1)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XShape = opset17.Shape(
        X)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Rank = opset17.Size(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XShape)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Zero1D = opset17.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[0]))
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Axis1D = opset17.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_PrefixShape = opset17.Slice(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XShape, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Zero1D, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Axis1D)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_NumReducedAxes = opset17.Sub(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Rank, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Axis1D)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_SuffixShape = opset17.ConstantOfShape(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_NumReducedAxes, value=make_tensor("value", 7, dims=[1], vals=[1]))
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_ReducedShape = opset17.Concat(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_PrefixShape, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_SuffixShape, axis=0)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_X2D = opset17.Flatten(
        X, axis=1)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XU = opset17.Cast(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_X2D, to=1)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Mean2D = opset17.ReduceMean(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XU, axes=[1])
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Square = opset17.Mul(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XU, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XU)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_MeanOfSquare = opset17.ReduceMean(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Square, axes=[1])
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_SquareOfMean = opset17.Mul(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Mean2D, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Mean2D)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Var = opset17.Sub(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_MeanOfSquare, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_SquareOfMean)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_VarPlusEpsilon = opset17.Add(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Var, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Epsilon)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_StdDev = opset17.Sqrt(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_VarPlusEpsilon)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Deviation = opset17.Sub(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XU, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Mean2D)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Normalized = opset17.Div(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Deviation, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_StdDev)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_NormalizedT = opset17.Cast(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Normalized, to=1)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Scale2D = opset17.Flatten(
        W, axis=0)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Scaled = opset17.Mul(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_NormalizedT, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Scale2D)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_B2D = opset17.Flatten(
        B, axis=0)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Biased = opset17.Add(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Scaled, LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_B2D)
    Y = opset17.Reshape(LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Biased,
                        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_XShape)
    LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_InvStdDev2D = opset17.Reciprocal(
        LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_StdDev)
    Mean = opset17.Reshape(LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_Mean2D,
                           LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_ReducedShape)
    InvStdDev = opset17.Reshape(LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_InvStdDev2D,
                                LayerNormalization_test_layer_normalization_4d_axis1_expanded_function_ReducedShape)
    return Y, Mean, InvStdDev
