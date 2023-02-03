
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_matmul_2d(a: FLOAT[3, 4], b: FLOAT[4, 3]) -> (FLOAT[3, 3]):

    c = opset13.MatMul(a, b)
    return c
