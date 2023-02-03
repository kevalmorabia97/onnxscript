
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_slice_neg(x: FLOAT[20, 10, 5], starts: INT64[1], ends: INT64[1], axes: INT64[1], steps: INT64[1]) -> (FLOAT[20, 9, 5]):

    y = opset13.Slice(x, starts, ends, axes, steps)
    return y
