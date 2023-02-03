
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, UINT8
from onnxscript.onnx_opset import opset10


@script()
def bck_test_qlinearconv(x: UINT8[1, 1, 7, 7], x_scale: FLOAT, x_zero_point: UINT8, w: UINT8[1, 1, 1, 1], w_scale: FLOAT[1], w_zero_point: UINT8[1], y_scale: FLOAT, y_zero_point: UINT8) -> (UINT8[1, 1, 7, 7]):

    y = opset10.QLinearConv(x, x_scale, x_zero_point, w,
                            w_scale, w_zero_point, y_scale, y_zero_point)
    return y
