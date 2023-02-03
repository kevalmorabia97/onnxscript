
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset12


@script()
def bck_test_celu(X: FLOAT[3, 3, 3, 1]) -> (FLOAT[3, 3, 3, 1]):

    Y = opset12.Celu(X, alpha=2.0)
    return Y
