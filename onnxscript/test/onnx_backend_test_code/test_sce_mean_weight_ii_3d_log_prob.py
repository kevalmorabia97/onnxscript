
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_sce_mean_weight_ii_3d_log_prob(x: FLOAT[3, 5, 2], y: INT64[3, 2], w: FLOAT[5]) -> (FLOAT, FLOAT[3, 5, 2]):

    z, log_prob = opset13.SoftmaxCrossEntropyLoss(
        x, y, w, ignore_index=1, reduction='mean')
    return z, log_prob
