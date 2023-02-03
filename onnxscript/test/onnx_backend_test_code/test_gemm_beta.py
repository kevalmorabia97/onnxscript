
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gemm_beta(a: FLOAT[2, 7], b: FLOAT[7, 4], c: FLOAT[1, 4]) -> (FLOAT[2, 4]):

    y = opset13.Gemm(a, b, c, beta=0.5)
    return y
