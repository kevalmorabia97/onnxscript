
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_nllloss_NCd1d2d3_sum_weight_high_ii(input: FLOAT[3, 5], target: INT64[3], weight: FLOAT[5]) -> (FLOAT):

    loss = opset13.NegativeLogLikelihoodLoss(
        input, target, weight, ignore_index=10, reduction='sum')
    return loss
