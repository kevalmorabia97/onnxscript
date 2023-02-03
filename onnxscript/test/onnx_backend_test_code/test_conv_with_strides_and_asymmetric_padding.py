
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_conv_with_strides_and_asymmetric_padding(x: FLOAT[1, 1, 7, 5], W: FLOAT[1, 1, 3, 3]) -> (FLOAT[1, 1, 4, 2]):

    y = opset11.Conv(x, W, kernel_shape=[3, 3], pads=[
                     1, 0, 1, 0], strides=[2, 2])
    return y
