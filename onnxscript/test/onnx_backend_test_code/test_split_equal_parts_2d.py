
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_split_equal_parts_2d(input: FLOAT[2, 6]) -> (FLOAT[2, 3], FLOAT[2, 3]):

    output_1, output_2 = opset13.Split(input, axis=1)
    return output_1, output_2
