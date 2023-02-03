
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_reduce_l1_negative_axes_keep_dims_example(data: FLOAT[3, 2, 2]) -> (FLOAT[3, 2, 1]):

    reduced = opset13.ReduceL1(data, axes=[-1], keepdims=1)
    return reduced
