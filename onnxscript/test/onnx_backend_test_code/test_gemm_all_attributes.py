
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gemm_all_attributes(a: FLOAT[4, 3], b: FLOAT[5, 4], c: FLOAT[1, 5]) -> (FLOAT[3, 5]):

    y = opset13.Gemm(a, b, c, alpha=0.25,
                     beta=0.3499999940395355, transA=1, transB=1)
    return y
