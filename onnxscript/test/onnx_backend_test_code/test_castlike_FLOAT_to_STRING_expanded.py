
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, STRING
from onnxscript.onnx_opset import opset15


@script()
def bck_test_castlike_FLOAT_to_STRING_expanded(input: FLOAT[3, 4], like: STRING[1]) -> (STRING[3, 4]):

    output = opset15.Cast(input, to=8)
    return output
