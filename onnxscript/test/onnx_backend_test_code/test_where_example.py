
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_where_example(condition: BOOL[2, 2], x: FLOAT[2, 2], y: FLOAT[2, 2]) -> (FLOAT[2, 2]):

    z = opset16.Where(condition, x, y)
    return z
