
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT8
from onnxscript.onnx_opset import opset13


@script()
def bck_test_clip_default_int8_inbounds(x: INT8[3]) -> (INT8[3]):

    y = opset13.Clip(x, None, None)
    return y
