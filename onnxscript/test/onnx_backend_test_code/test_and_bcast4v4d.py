
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL
from onnxscript.onnx_opset import opset7


@script()
def bck_test_and_bcast4v4d(x: BOOL[1, 4, 1, 6], y: BOOL[3, 1, 5, 6]) -> (BOOL[3, 4, 5, 6]):

    r_and = opset7.And(x, y)
    return r_and
