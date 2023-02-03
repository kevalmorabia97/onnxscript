
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_mixed_sign_int32(x: INT32[6], y: INT32[6]) -> (INT32[6]):

    z = opset13.Mod(x, y)
    return z
