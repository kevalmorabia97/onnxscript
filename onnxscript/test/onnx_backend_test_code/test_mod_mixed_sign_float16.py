
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT16
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mod_mixed_sign_float16(x: FLOAT16[6], y: FLOAT16[6]) -> (FLOAT16[6]):

    z = opset13.Mod(x, y, fmod=1)
    return z
