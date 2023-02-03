
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_concat_2d_axis_0(value0: FLOAT[2, 2], value1: FLOAT[2, 2]) -> (FLOAT[4, 2]):

    output = opset13.Concat(value0, value1, axis=0)
    return output
