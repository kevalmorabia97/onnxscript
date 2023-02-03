
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_averagepool_2d_ceil(x: FLOAT[1, 1, 4, 4]) -> (FLOAT[1, 1, 2, 2]):

    y = opset11.AveragePool(x, ceil_mode=1, kernel_shape=[
                            3, 3], strides=[2, 2])
    return y
