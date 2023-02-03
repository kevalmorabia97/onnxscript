
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_lrn_default(x: FLOAT[5, 5, 5, 5]) -> (FLOAT[5, 5, 5, 5]):

    y = opset13.LRN(x, size=3)
    return y
