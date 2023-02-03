
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_convtranspose_dilations(X: FLOAT[1, 1, 3, 3], W: FLOAT[1, 1, 2, 2]) -> (FLOAT[1, 1, 5, 5]):

    Y = opset11.ConvTranspose(X, W, dilations=[2, 2])
    return Y
