
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_nllloss_NCd1d2d3_none_no_weight_negative_ii(input: FLOAT[3, 5, 6, 6, 5], target: INT64[3, 6, 6, 5]) -> (FLOAT[3, 6, 6, 5]):

    loss = opset13.NegativeLogLikelihoodLoss(
        input, target, ignore_index=-5, reduction='none')
    return loss
