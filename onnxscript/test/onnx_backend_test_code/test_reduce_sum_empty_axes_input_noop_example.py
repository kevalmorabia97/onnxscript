
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_reduce_sum_empty_axes_input_noop_example(data: FLOAT[3, 2, 2], axes: INT64[0]) -> (FLOAT[3, 2, 2]):

    reduced = opset13.ReduceSum(data, axes, keepdims=1, noop_with_empty_axes=1)
    return reduced
