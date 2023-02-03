
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import UINT16
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_uint16(x: UINT16[3], y: UINT16[3]) -> (UINT16[3]):

    z = opset13.Mod(x, y)
    return z
