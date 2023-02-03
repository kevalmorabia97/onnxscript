
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset6


@script()
def bck_test_selu_example(x: FLOAT[3]) -> (FLOAT[3]):

    y = opset6.Selu(x, alpha=2.0, gamma=3.0)
    return y
