
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset13


@script()
def bck_test_reduce_log_sum_exp_do_not_keepdims_random(data: DOUBLE[3, 2, 2]) -> (DOUBLE[3, 2]):

    reduced = opset13.ReduceLogSumExp(data, axes=[1], keepdims=0)
    return reduced
