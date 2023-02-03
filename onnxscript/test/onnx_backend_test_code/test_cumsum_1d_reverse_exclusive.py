
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE, INT32
from onnxscript.onnx_opset import opset14


@script()
def bck_test_cumsum_1d_reverse_exclusive(x: DOUBLE[5], axis: INT32) -> (DOUBLE[5]):

    y = opset14.CumSum(x, axis, exclusive=1, reverse=1)
    return y
