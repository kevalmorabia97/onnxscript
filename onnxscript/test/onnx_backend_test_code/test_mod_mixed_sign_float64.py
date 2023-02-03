
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_mixed_sign_float64(x: DOUBLE[6], y: DOUBLE[6]) -> (DOUBLE[6]):

    z = opset13.Mod(x, y, fmod=1)
    return z
