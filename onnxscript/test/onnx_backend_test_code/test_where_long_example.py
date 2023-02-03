
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, INT64
from onnxscript.onnx_opset import opset16


@script()
def bck_test_where_long_example(condition: BOOL[2, 2], x: INT64[2, 2], y: INT64[2, 2]) -> (INT64[2, 2]):

    z = opset16.Where(condition, x, y)
    return z
