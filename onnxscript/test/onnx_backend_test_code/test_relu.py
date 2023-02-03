
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset14


@script()
def bck_test_relu(x: FLOAT[3, 4, 5]) -> (FLOAT[3, 4, 5]):

    y = opset14.Relu(x)
    return y
