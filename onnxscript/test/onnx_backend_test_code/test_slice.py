
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_slice(x: FLOAT[20, 10, 5], starts: INT64[2], ends: INT64[2], axes: INT64[2], steps: INT64[2]) -> (FLOAT[3, 10, 5]):

    y = opset13.Slice(x, starts, ends, axes, steps)
    return y
