
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_unsqueeze_negative_axes(x: FLOAT[1, 3, 1, 5], axes: INT64[1]) -> (FLOAT[1, 3, 1, 1, 5]):

    y = opset13.Unsqueeze(x, axes)
    return y
