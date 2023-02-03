
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE, FLOAT
from onnxscript.onnx_opset import opset15


@script()
def bck_test_bernoulli_double(x: FLOAT[10]) -> (DOUBLE[10]):

    y = opset15.Bernoulli(x, dtype=11)
    return y
