
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15


@script()
def bck_test_bernoulli_seed(x: FLOAT[10]) -> (FLOAT[10]):

    y = opset15.Bernoulli(x, seed=0.0)
    return y
