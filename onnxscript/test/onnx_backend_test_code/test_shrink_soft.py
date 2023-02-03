
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset9


@script()
def bck_test_shrink_soft(x: FLOAT[5]) -> (FLOAT[5]):

    y = opset9.Shrink(x, bias=1.5, lambd=1.5)
    return y
