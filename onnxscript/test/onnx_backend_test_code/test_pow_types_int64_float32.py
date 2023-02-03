
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset15


@script()
def bck_test_pow_types_int64_float32(x: INT64[3], y: FLOAT[3]) -> (INT64[3]):

    z = opset15.Pow(x, y)
    return z
