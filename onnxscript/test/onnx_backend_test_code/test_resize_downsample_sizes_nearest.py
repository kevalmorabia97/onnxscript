
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_resize_downsample_sizes_nearest(X: FLOAT[1, 1, 2, 4], sizes: INT64[4]) -> (FLOAT[1, 1, 1, 3]):

    Y = opset13.Resize(X, None, None, sizes, mode='nearest')
    return Y
