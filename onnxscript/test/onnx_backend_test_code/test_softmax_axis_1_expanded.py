
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_softmax_axis_1_expanded(x: FLOAT[3, 4, 5]) -> (FLOAT[3, 4, 5]):

    Softmax_test_softmax_axis_1_expanded_function_axes = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    Softmax_test_softmax_axis_1_expanded_function_X_ReduceMax = opset13.ReduceMax(
        x, keepdims=1, axes=[1])
    Softmax_test_softmax_axis_1_expanded_function_X_Sub = opset13.Sub(
        x, Softmax_test_softmax_axis_1_expanded_function_X_ReduceMax)
    Softmax_test_softmax_axis_1_expanded_function_X_Exp = opset13.Exp(
        Softmax_test_softmax_axis_1_expanded_function_X_Sub)
    Softmax_test_softmax_axis_1_expanded_function_X_ReduceSum = opset13.ReduceSum(
        Softmax_test_softmax_axis_1_expanded_function_X_Exp, Softmax_test_softmax_axis_1_expanded_function_axes, keepdims=1)
    y = opset13.Div(Softmax_test_softmax_axis_1_expanded_function_X_Exp,
                    Softmax_test_softmax_axis_1_expanded_function_X_ReduceSum)
    return y
