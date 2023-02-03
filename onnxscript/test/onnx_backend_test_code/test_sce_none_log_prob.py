
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_sce_none_log_prob(x: FLOAT[3, 5], y: INT64[3]) -> (FLOAT[3], FLOAT[3, 5]):

    z, log_prob = opset13.SoftmaxCrossEntropyLoss(x, y, reduction='none')
    return z, log_prob
