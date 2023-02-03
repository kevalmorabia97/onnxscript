
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE, FLOAT16
from onnxscript.onnx_opset import opset13


@script()
def bck_test_cast_DOUBLE_to_FLOAT16(input: DOUBLE[3, 4]) -> (FLOAT16[3, 4]):

    output = opset13.Cast(input, to=10)
    return output
