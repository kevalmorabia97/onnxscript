
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_reduce_log_sum_asc_axes(data: FLOAT[3, 4, 5]) -> (FLOAT[5]):

    reduced = opset13.ReduceLogSum(data, axes=[0, 1], keepdims=0)
    return reduced
