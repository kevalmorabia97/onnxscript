
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_identity(x: FLOAT[1, 1, 2, 2]) -> (FLOAT[1, 1, 2, 2]):

    y = opset16.Identity(x)
    return y
