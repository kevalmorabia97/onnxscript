
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_nllloss_NCd1d2d3d4d5_none_no_weight(input: FLOAT[3, 5, 6, 6, 5, 3, 4], target: INT64[3, 6, 6, 5, 3, 4]) -> (FLOAT[3, 6, 6, 5, 3, 4]):

    loss = opset13.NegativeLogLikelihoodLoss(input, target, reduction='none')
    return loss
