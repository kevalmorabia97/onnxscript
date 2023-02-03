
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, UINT8
from onnxscript.onnx_opset import opset11


@script()
def bck_test_dynamicquantizelinear_max_adjusted_expanded(x: FLOAT[6]) -> (UINT8[6], FLOAT, UINT8):

    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Min = opset11.Constant(
        value=make_tensor("value", 1, dims=[], vals=[0.0]))
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Max = opset11.Constant(
        value=make_tensor("value", 1, dims=[], vals=[255.0]))
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Min = opset11.ReduceMin(
        x, keepdims=0)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Min_Adjusted = opset11.Min(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Min, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Min)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Max = opset11.ReduceMax(
        x, keepdims=0)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Max_Adjusted = opset11.Max(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Max, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Min)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Range = opset11.Sub(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Max_Adjusted, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Min_Adjusted)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Scale = opset11.Div(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Range, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Max)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Min_Scaled = opset11.Div(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_X_Min_Adjusted, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Scale)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Initial_ZeroPoint_FP = opset11.Sub(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Min, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Min_Scaled)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Clipped_ZeroPoint_FP = opset11.Clip(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Initial_ZeroPoint_FP, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Min, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Q_Max)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Rounded_ZeroPoint_FP = opset11.Round(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Clipped_ZeroPoint_FP)
    DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Zeropoint = opset11.Cast(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Rounded_ZeroPoint_FP, to=2)
    y_scale = opset11.Identity(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Scale)
    y_zero_point = opset11.Identity(
        DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Zeropoint)
    y = opset11.QuantizeLinear(x, DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Scale,
                               DynamicQuantizeLinear_test_dynamicquantizelinear_max_adjusted_expanded_function_Zeropoint)
    return y, y_scale, y_zero_point
