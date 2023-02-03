
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_less_equal(x: FLOAT[3, 4, 5], y: FLOAT[3, 4, 5]) -> (BOOL[3, 4, 5]):

    less_equal = opset16.LessOrEqual(x, y)
    return less_equal
