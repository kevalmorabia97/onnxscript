
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_slice_negative_axes(x: FLOAT[20, 10, 5], starts: INT64[3], ends: INT64[3], axes: INT64[3]) -> (FLOAT[20, 10, 1]):

    y = opset13.Slice(x, starts, ends, axes)
    return y
