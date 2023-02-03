
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_flatten_axis1(a: FLOAT[2, 3, 4, 5]) -> (FLOAT[2, 60]):

    b = opset13.Flatten(a, axis=1)
    return b
