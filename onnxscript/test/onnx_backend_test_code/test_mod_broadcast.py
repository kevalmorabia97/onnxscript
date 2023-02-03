
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_broadcast(x: INT32[3, 2, 5], y: INT32[1]) -> (INT32[3, 2, 5]):

    z = opset13.Mod(x, y)
    return z
