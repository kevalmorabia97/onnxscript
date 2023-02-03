
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_argmin_default_axis_example_select_last_index(data: FLOAT[2, 2]) -> (INT64[1, 2]):

    result = opset13.ArgMin(data, keepdims=1, select_last_index=1)
    return result
