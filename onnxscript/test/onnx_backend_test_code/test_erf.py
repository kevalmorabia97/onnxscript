
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_erf(x: FLOAT[1, 3, 32, 32]) -> (FLOAT[1, 3, 32, 32]):

    y = opset13.Erf(x)
    return y
