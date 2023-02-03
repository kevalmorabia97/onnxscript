
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset15


@script()
def bck_test_shape_clip_end(x: FLOAT[3, 4, 5]) -> (INT64[3]):

    y = opset15.Shape(x, end=10)
    return y
