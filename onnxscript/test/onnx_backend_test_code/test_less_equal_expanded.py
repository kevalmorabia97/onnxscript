
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_less_equal_expanded(x: FLOAT[3, 4, 5], y: FLOAT[3, 4, 5]) -> (BOOL[3, 4, 5]):

    LessOrEqual_test_less_equal_expanded_function_O1 = opset16.Less(x, y)
    LessOrEqual_test_less_equal_expanded_function_O2 = opset16.Equal(x, y)
    less_equal = opset16.Or(LessOrEqual_test_less_equal_expanded_function_O1,
                            LessOrEqual_test_less_equal_expanded_function_O2)
    return less_equal
