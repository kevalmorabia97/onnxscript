
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL
from onnxscript.onnx_opset import opset7


@script()
def bck_test_xor_bcast4v2d(x: BOOL[3, 4, 5, 6], y: BOOL[5, 6]) -> (BOOL[3, 4, 5, 6]):

    xor = opset7.Xor(x, y)
    return xor
