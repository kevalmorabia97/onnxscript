
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset12


@script()
def bck_test_maxpool_2d_precomputed_strides(x: FLOAT[1, 1, 5, 5]) -> (FLOAT[1, 1, 2, 2]):

    y = opset12.MaxPool(x, kernel_shape=[2, 2], strides=[2, 2])
    return y
