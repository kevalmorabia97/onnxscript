
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset12


@script()
def bck_test_einsum_transpose(x: DOUBLE[3, 4]) -> (DOUBLE[4, 3]):

    y = opset12.Einsum(x, equation='ij->ji')
    return y
