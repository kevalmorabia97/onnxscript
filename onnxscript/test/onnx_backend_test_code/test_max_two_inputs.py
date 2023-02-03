
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_max_two_inputs(data_0: FLOAT[3], data_1: FLOAT[3]) -> (FLOAT[3]):

    result = opset13.Max(data_0, data_1)
    return result
