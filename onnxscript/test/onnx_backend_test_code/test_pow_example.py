
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15


@script()
def bck_test_pow_example(x: FLOAT[3], y: FLOAT[3]) -> (FLOAT[3]):

    z = opset15.Pow(x, y)
    return z
