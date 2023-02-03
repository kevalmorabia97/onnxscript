
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_argmax_no_keepdims_random(data: FLOAT[2, 3, 4]) -> (INT64[2, 4]):

    result = opset13.ArgMax(data, axis=1, keepdims=0)
    return result
