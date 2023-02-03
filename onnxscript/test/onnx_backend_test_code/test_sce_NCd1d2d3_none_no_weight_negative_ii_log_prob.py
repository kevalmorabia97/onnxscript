
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob(x: FLOAT[3, 5, 6, 6, 5], y: INT64[3, 6, 6, 5]) -> (FLOAT[3, 6, 6, 5], FLOAT[3, 5, 6, 6, 5]):

    z, log_prob = opset13.SoftmaxCrossEntropyLoss(
        x, y, ignore_index=-5, reduction='none')
    return z, log_prob
