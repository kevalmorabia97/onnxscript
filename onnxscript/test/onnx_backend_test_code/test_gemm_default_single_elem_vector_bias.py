
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gemm_default_single_elem_vector_bias(a: FLOAT[3, 7], b: FLOAT[7, 3], c: FLOAT[1]) -> (FLOAT[3, 3]):

    y = opset13.Gemm(a, b, c)
    return y
