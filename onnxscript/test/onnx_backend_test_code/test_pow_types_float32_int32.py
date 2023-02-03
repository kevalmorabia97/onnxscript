
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.onnx_opset import opset15


@script()
def bck_test_pow_types_float32_int32(x: FLOAT[3], y: INT32[3]) -> (FLOAT[3]):

    z = opset15.Pow(x, y)
    return z
