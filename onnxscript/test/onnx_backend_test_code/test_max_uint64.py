
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import UINT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_max_uint64(data_0: UINT64[3], data_1: UINT64[3]) -> (UINT64[3]):

    result = opset13.Max(data_0, data_1)
    return result
