
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset12


@script()
def bck_test_einsum_batch_diagonal(x: DOUBLE[3, 5, 5]) -> (DOUBLE[3, 5]):

    y = opset12.Einsum(x, equation='...ii ->...i')
    return y
