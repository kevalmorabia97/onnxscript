
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE
from onnxscript.onnx_opset import opset12


@script()
def bck_test_einsum_batch_matmul(x: DOUBLE[5, 2, 3], y: DOUBLE[5, 3, 4]) -> (DOUBLE[5, 2, 4]):

    z = opset12.Einsum(x, y, equation='bij, bjk -> bik')
    return z
