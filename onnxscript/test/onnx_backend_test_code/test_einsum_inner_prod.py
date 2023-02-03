
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset12


@script()
def bck_test_einsum_inner_prod(x: DOUBLE[5], y: DOUBLE[5]) -> (DOUBLE):

    z = opset12.Einsum(x, y, equation='i,i')
    return z
