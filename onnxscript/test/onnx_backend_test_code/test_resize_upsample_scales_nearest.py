
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_resize_upsample_scales_nearest(X: FLOAT[1, 1, 2, 2], scales: FLOAT[4]) -> (FLOAT[1, 1, 4, 6]):

    Y = opset13.Resize(X, None, scales, mode='nearest')
    return Y
