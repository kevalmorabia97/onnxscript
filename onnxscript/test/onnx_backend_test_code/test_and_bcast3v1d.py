
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL
from onnxscript.onnx_opset import opset7


@script()
def bck_test_and_bcast3v1d(x: BOOL[3, 4, 5], y: BOOL[5]) -> (BOOL[3, 4, 5]):

    r_and = opset7.And(x, y)
    return r_and
