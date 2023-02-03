
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_clip_default_max(x: FLOAT[3, 4, 5], max: FLOAT) -> (FLOAT[3, 4, 5]):

    y = opset13.Clip(x, None, max)
    return y
