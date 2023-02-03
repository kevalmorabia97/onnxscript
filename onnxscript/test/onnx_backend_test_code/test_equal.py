
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, INT32
from onnxscript.onnx_opset import opset13


@script()
def bck_test_equal(x: INT32[3, 4, 5], y: INT32[3, 4, 5]) -> (BOOL[3, 4, 5]):

    z = opset13.Equal(x, y)
    return z
