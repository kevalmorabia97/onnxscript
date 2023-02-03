
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_greater_equal_bcast_expanded(x: FLOAT[3, 4, 5], y: FLOAT[5]) -> (BOOL[3, 4, 5]):

    GreaterOrEqual_test_greater_equal_bcast_expanded_function_O1 = opset16.Greater(
        x, y)
    GreaterOrEqual_test_greater_equal_bcast_expanded_function_O2 = opset16.Equal(
        x, y)
    greater_equal = opset16.Or(GreaterOrEqual_test_greater_equal_bcast_expanded_function_O1,
                               GreaterOrEqual_test_greater_equal_bcast_expanded_function_O2)
    return greater_equal
