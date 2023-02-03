
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_unsqueeze_two_axes(x: FLOAT[3, 4, 5], axes: INT64[2]) -> (FLOAT[3, 1, 4, 5, 1]):

    y = opset13.Unsqueeze(x, axes)
    return y
