
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_argmax_default_axis_example(data: FLOAT[2, 2]) -> (INT64[1, 2]):

    result = opset13.ArgMax(data, keepdims=1)
    return result
