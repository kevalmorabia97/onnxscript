
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT16
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_mixed_sign_int16(x: INT16[6], y: INT16[6]) -> (INT16[6]):

    z = opset13.Mod(x, y)
    return z
