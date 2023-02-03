
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_onehot_without_axis(indices: INT64[3], depth: FLOAT, values: INT32[2]) -> (INT32[3, 12]):

    y = opset11.OneHot(indices, depth, values)
    return y
