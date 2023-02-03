
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_sce_mean_3d(x: FLOAT[3, 5, 2], y: INT64[3, 2]) -> (FLOAT):

    z = opset13.SoftmaxCrossEntropyLoss(x, y, reduction='mean')
    return z
