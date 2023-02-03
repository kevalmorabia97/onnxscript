
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import UINT32
from onnxscript.onnx_opset import opset13


@script()
def bck_test_min_uint32(data_0: UINT32[3], data_1: UINT32[3]) -> (UINT32[3]):

    result = opset13.Min(data_0, data_1)
    return result
