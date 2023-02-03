
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32
from onnxscript.onnx_opset import opset11


@script()
def bck_test_range_int32_type_negative_delta(start: INT32, limit: INT32, delta: INT32) -> (INT32[2]):

    output = opset11.Range(start, limit, delta)
    return output
