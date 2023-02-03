
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_mixed_sign_float32(x: FLOAT[6], y: FLOAT[6]) -> (FLOAT[6]):

    z = opset13.Mod(x, y, fmod=1)
    return z
