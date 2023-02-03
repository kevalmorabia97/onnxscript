
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, UINT8
from onnxscript.onnx_opset import opset11


@script()
def bck_test_dynamicquantizelinear_min_adjusted(x: FLOAT[3, 4]) -> (UINT8[3, 4], FLOAT, UINT8):

    y, y_scale, y_zero_point = opset11.DynamicQuantizeLinear(x)
    return y, y_scale, y_zero_point
