
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL
from onnxscript.onnx_opset import opset7


@script()
def bck_test_or2d(x: BOOL[3, 4], y: BOOL[3, 4]) -> (BOOL[3, 4]):

    r_or = opset7.Or(x, y)
    return r_or
