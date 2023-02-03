
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_sce_NCd1d2d3_sum_weight_high_ii_expanded(x: FLOAT[3, 5], y: INT64[3], w: FLOAT[5]) -> (FLOAT):

    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_Shape3D = opset13.Constant(
        value=make_tensor("value", 7, dims=[3], vals=[0, 0, -1]))
    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_NCD = opset13.Reshape(
        x, SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_Shape3D)
    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_NDC = opset13.Transpose(
        SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_NCD, perm=[0, 2, 1])
    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_LogSM = opset13.LogSoftmax(
        SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_NDC, axis=2)
    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_LogSM_NCD = opset13.Transpose(
        SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_LogSM, perm=[0, 2, 1])
    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_shape = opset13.Shape(
        x)
    SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_Log = opset13.Reshape(
        SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_LogSM_NCD, SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_shape)
    z = opset13.NegativeLogLikelihoodLoss(
        SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_sum_weight_high_ii_expanded_function_X_Log, y, w, reduction='sum', ignore_index=10)
    return z
