
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_lrn(x: FLOAT[5, 5, 5, 5]) -> (FLOAT[5, 5, 5, 5]):

    y = opset13.LRN(x, alpha=0.00019999999494757503,
                    beta=0.5, bias=2.0, size=3)
    return y
