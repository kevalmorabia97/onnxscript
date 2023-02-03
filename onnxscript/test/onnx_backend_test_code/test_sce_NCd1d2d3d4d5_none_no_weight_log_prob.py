
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_sce_NCd1d2d3d4d5_none_no_weight_log_prob(x: FLOAT[3, 5, 6, 6, 5, 3, 4], y: INT64[3, 6, 6, 5, 3, 4]) -> (FLOAT[3, 6, 6, 5, 3, 4], FLOAT[3, 5, 6, 6, 5, 3, 4]):

    z, log_prob = opset13.SoftmaxCrossEntropyLoss(x, y, reduction='none')
    return z, log_prob
