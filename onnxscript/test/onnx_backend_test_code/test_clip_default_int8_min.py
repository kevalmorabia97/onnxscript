
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT8
from onnxscript.onnx_opset import opset13


@script()
def bck_test_clip_default_int8_min(x: INT8[3, 4, 5], min: INT8) -> (INT8[3, 4, 5]):

    y = opset13.Clip(x, min)
    return y
