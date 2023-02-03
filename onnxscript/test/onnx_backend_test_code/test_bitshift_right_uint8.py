
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import UINT8
from onnxscript.onnx_opset import opset11


@script()
def bck_test_bitshift_right_uint8(x: UINT8[3], y: UINT8[3]) -> (UINT8[3]):

    z = opset11.BitShift(x, y, direction='RIGHT')
    return z
