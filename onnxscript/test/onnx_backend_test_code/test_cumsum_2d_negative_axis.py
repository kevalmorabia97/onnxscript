
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE, INT32
from onnxscript.onnx_opset import opset14


@script()
def bck_test_cumsum_2d_negative_axis(x: DOUBLE[2, 3], axis: INT32) -> (DOUBLE[2, 3]):

    y = opset14.CumSum(x, axis)
    return y
