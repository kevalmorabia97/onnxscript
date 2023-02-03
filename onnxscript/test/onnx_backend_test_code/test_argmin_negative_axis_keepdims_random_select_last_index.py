
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_argmin_negative_axis_keepdims_random_select_last_index(data: FLOAT[2, 3, 4]) -> (INT64[2, 3, 1]):

    result = opset13.ArgMin(data, axis=-1, keepdims=1, select_last_index=1)
    return result
