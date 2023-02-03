
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_nllloss_NCd1_expanded(input: FLOAT[3, 5, 2], target: INT64[3, 2]) -> (FLOAT):

    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_const_zero = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[0]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_const_one = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_axes = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_expanded_target = opset13.Unsqueeze(
        target, NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_axes)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_input_gather_element = opset13.GatherElements(
        input, NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_expanded_target, axis=1)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_loss_NCdd = opset13.Neg(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_input_gather_element)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_loss_N1dd = opset13.Slice(NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_loss_NCdd, NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_const_zero,
                                                                                            NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_const_one, NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_const_one)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_loss_Ndd = opset13.Squeeze(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_loss_N1dd, NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_axes)
    loss = opset13.ReduceMean(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_expanded_function_loss_Ndd, keepdims=0)
    return loss
