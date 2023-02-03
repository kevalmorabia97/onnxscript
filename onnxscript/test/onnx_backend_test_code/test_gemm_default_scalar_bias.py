
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gemm_default_scalar_bias(a: FLOAT[2, 3], b: FLOAT[3, 4], c: FLOAT) -> (FLOAT[2, 4]):

    y = opset13.Gemm(a, b, c)
    return y
