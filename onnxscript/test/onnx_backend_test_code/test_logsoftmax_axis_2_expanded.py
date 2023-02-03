
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_logsoftmax_axis_2_expanded(x: FLOAT[3, 4, 5]) -> (FLOAT[3, 4, 5]):

    LogSoftmax_test_logsoftmax_axis_2_expanded_function_axes = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[2]))
    LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_ReduceMax = opset13.ReduceMax(
        x, keepdims=1, axes=[2])
    LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Sub = opset13.Sub(
        x, LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_ReduceMax)
    LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Exp = opset13.Exp(
        LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Sub)
    LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_ReduceSum = opset13.ReduceSum(
        LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Exp, LogSoftmax_test_logsoftmax_axis_2_expanded_function_axes, keepdims=1)
    LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Log = opset13.Log(
        LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_ReduceSum)
    y = opset13.Sub(LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Sub,
                    LogSoftmax_test_logsoftmax_axis_2_expanded_function_X_Log)
    return y
