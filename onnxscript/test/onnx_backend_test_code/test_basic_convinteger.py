
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32, UINT8
from onnxscript.onnx_opset import opset10


@script()
def bck_test_basic_convinteger(x: UINT8[1, 1, 3, 3], w: UINT8[1, 1, 2, 2], x_zero_point: UINT8) -> (INT32[1, 1, 2, 2]):

    y = opset10.ConvInteger(x, w, x_zero_point)
    return y
