
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset14


@script()
def bck_test_div_bcast(x: FLOAT[3, 4, 5], y: FLOAT[5]) -> (FLOAT[3, 4, 5]):

    z = opset14.Div(x, y)
    return z
