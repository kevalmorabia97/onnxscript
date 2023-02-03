
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_convtranspose_pad(X: FLOAT[1, 1, 3, 3], W: FLOAT[1, 2, 3, 3]) -> (FLOAT[1, 2, 10, 8]):

    Y = opset11.ConvTranspose(X, W, output_padding=[1, 1], strides=[3, 2])
    return Y
