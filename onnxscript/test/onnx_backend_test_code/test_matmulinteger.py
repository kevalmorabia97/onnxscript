
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32, UINT8
from onnxscript.onnx_opset import opset10


@script()
def bck_test_matmulinteger(A: UINT8[4, 3], B: UINT8[3, 2], a_zero_point: UINT8[1], b_zero_point: UINT8[1]) -> (INT32[4, 2]):

    Y = opset10.MatMulInteger(A, B, a_zero_point, b_zero_point)
    return Y
