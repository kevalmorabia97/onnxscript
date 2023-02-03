
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import UINT8
from onnxscript.onnx_opset import opset12


@script()
def bck_test_maxpool_2d_uint8(x: UINT8[1, 1, 5, 5]) -> (UINT8[1, 1, 5, 5]):

    y = opset12.MaxPool(x, kernel_shape=[5, 5], pads=[2, 2, 2, 2])
    return y
