
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_split_equal_parts_1d(input: FLOAT[6]) -> (FLOAT[2], FLOAT[2], FLOAT[2]):

    output_1, output_2, output_3 = opset13.Split(input, axis=0)
    return output_1, output_2, output_3
