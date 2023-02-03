
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gemm_default_no_bias(a: FLOAT[2, 10], b: FLOAT[10, 3]) -> (FLOAT[2, 3]):

    y = opset13.Gemm(a, b)
    return y
