
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_prelu_broadcast(x: FLOAT[3, 4, 5], slope: FLOAT[5]) -> (FLOAT[3, 4, 5]):

    y = opset16.PRelu(x, slope)
    return y
