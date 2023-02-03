
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_tile(x: FLOAT[2, 3, 4, 5], y: INT64[4]) -> (FLOAT[14, 18, 16, 10]):

    z = opset13.Tile(x, y)
    return z
