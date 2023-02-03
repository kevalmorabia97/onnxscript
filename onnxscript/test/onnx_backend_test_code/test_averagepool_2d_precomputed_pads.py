
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_averagepool_2d_precomputed_pads(x: FLOAT[1, 1, 5, 5]) -> (FLOAT[1, 1, 5, 5]):

    y = opset11.AveragePool(x, kernel_shape=[5, 5], pads=[2, 2, 2, 2])
    return y
