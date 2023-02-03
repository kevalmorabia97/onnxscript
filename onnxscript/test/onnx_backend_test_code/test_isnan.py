
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_isnan(x: FLOAT[4]) -> (BOOL[4]):

    y = opset13.IsNaN(x)
    return y
