
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_flatten_default_axis(a: FLOAT[5, 4, 3, 2]) -> (FLOAT[5, 24]):

    b = opset13.Flatten(a)
    return b
