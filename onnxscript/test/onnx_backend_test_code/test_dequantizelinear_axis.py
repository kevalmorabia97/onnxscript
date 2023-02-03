
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, UINT8
from onnxscript.onnx_opset import opset13


@script()
def bck_test_dequantizelinear_axis(x: UINT8[1, 3, 3, 2], x_scale: FLOAT[3], x_zero_point: UINT8[3]) -> (FLOAT[1, 3, 3, 2]):

    y = opset13.DequantizeLinear(x, x_scale, x_zero_point)
    return y
