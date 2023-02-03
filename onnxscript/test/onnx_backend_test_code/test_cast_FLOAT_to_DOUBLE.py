
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE, FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_cast_FLOAT_to_DOUBLE(input: FLOAT[3, 4]) -> (DOUBLE[3, 4]):

    output = opset13.Cast(input, to=11)
    return output
