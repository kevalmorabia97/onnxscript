
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT16
from onnxscript.onnx_opset import opset13


@script()
def bck_test_max_float16(data_0: FLOAT16[3], data_1: FLOAT16[3]) -> (FLOAT16[3]):

    result = opset13.Max(data_0, data_1)
    return result
