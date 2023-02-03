
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, UINT8
from onnxscript.onnx_opset import opset10


@script()
def bck_test_qlinearmatmul_2D(a: UINT8[2, 4], a_scale: FLOAT[1], a_zero_point: UINT8[1], b: UINT8[4, 3], b_scale: FLOAT[1], b_zero_point: UINT8[1], y_scale: FLOAT[1], y_zero_point: UINT8[1]) -> (UINT8[2, 3]):

    y = opset10.QLinearMatMul(a, a_scale, a_zero_point,
                              b, b_scale, b_zero_point, y_scale, y_zero_point)
    return y
