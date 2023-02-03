
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_reduce_prod_default_axes_keepdims_example(data: FLOAT[3, 2, 2]) -> (FLOAT[1, 1, 1]):

    reduced = opset13.ReduceProd(data, keepdims=1)
    return reduced
