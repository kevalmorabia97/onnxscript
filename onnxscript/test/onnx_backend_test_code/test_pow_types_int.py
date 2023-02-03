
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset12


@script()
def bck_test_pow_types_int(x: FLOAT[3], y: INT64[3]) -> (FLOAT[3]):

    z = opset12.Pow(x, y)
    return z
