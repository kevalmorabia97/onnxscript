
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_basic_conv_without_padding(x: FLOAT[1, 1, 5, 5], W: FLOAT[1, 1, 3, 3]) -> (FLOAT[1, 1, 3, 3]):

    y = opset11.Conv(x, W, kernel_shape=[3, 3], pads=[0, 0, 0, 0])
    return y
