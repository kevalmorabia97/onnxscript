
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL
from onnxscript.onnx_opset import opset7


@script()
def bck_test_or_bcast4v3d(x: BOOL[3, 4, 5, 6], y: BOOL[4, 5, 6]) -> (BOOL[3, 4, 5, 6]):

    r_or = opset7.Or(x, y)
    return r_or
