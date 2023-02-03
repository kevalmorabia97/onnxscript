
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_nllloss_NCd1_ii_expanded(input: FLOAT[3, 5, 2], target: INT64[3, 2]) -> (FLOAT):

    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[0]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_one = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_axes = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target = opset13.Unsqueeze(
        target, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_axes)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_ignore_index = opset13.Constant(
        value=make_tensor("value", 7, dims=[1], vals=[1]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero_target_typed = opset13.Sub(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target_int64 = opset13.Cast(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target, to=7)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_mask = opset13.Equal(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target_int64, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_ignore_index)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_transform_targets = opset13.Where(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_mask, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero_target_typed, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_expanded_target)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_input_gather_element = opset13.GatherElements(
        input, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_transform_targets, axis=1)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero_float = opset13.Constant(
        value=make_tensor("value", 1, dims=[1], vals=[0.0]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_input_gather_element_transform = opset13.Where(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_mask, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero_float, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_input_gather_element)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_NCdd = opset13.Neg(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_input_gather_element_transform)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_N1dd = opset13.Slice(NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_NCdd, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero,
                                                                                               NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_one, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_one)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_squeeze_mask = opset13.Squeeze(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_mask, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_axes)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_one_float = opset13.Constant(
        value=make_tensor("value", 1, dims=[1], vals=[1.0]))
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_weight_gather = opset13.Where(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_squeeze_mask, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_zero_float, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_const_one_float)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_unweighted = opset13.Squeeze(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_N1dd, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_axes)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_Ndd = opset13.Mul(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_unweighted, NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_weight_gather)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_sum = opset13.ReduceSum(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_Ndd, keepdims=0)
    NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_weight_gather_sum = opset13.ReduceSum(
        NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_weight_gather, keepdims=0)
    loss = opset13.Div(NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_loss_sum,
                       NegativeLogLikelihoodLoss_test_nllloss_NCd1_ii_expanded_function_weight_gather_sum)
    return loss
