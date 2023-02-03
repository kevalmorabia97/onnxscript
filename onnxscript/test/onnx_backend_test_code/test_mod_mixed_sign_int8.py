
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT8
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_mixed_sign_int8(x: INT8[6], y: INT8[6]) -> (INT8[6]):

    z = opset13.Mod(x, y)
    return z
