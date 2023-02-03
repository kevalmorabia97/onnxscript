
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, UINT8
from onnxscript.onnx_opset import opset13


@script()
def bck_test_quantizelinear(x: FLOAT[6], y_scale: FLOAT, y_zero_point: UINT8) -> (UINT8[6]):

    y = opset13.QuantizeLinear(x, y_scale, y_zero_point)
    return y
