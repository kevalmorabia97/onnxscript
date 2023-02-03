
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gemm_transposeA(a: FLOAT[6, 3], b: FLOAT[6, 4], c: FLOAT[1, 4]) -> (FLOAT[3, 4]):

    y = opset13.Gemm(a, b, c, transA=1)
    return y
