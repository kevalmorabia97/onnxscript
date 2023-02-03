
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset13


@script()
def bck_test_min_float64(data_0: DOUBLE[3], data_1: DOUBLE[3]) -> (DOUBLE[3]):

    result = opset13.Min(data_0, data_1)
    return result
