
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_averagepool_2d_precomputed_same_upper(x: FLOAT[1, 1, 5, 5]) -> (FLOAT[1, 1, 3, 3]):

    y = opset11.AveragePool(x, auto_pad='SAME_UPPER',
                            kernel_shape=[3, 3], strides=[2, 2])
    return y
