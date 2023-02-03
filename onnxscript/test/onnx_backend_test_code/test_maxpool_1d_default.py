
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset12


@script()
def bck_test_maxpool_1d_default(x: FLOAT[1, 3, 32]) -> (FLOAT[1, 3, 31]):

    y = opset12.MaxPool(x, kernel_shape=[2])
    return y
