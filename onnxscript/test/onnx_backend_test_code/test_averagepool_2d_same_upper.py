
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_averagepool_2d_same_upper(x: FLOAT[1, 3, 32, 32]) -> (FLOAT[1, 3, 32, 32]):

    y = opset11.AveragePool(x, auto_pad='SAME_UPPER', kernel_shape=[2, 2])
    return y
