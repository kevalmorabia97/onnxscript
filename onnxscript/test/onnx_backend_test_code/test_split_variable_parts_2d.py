
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_split_variable_parts_2d(input: FLOAT[2, 6], split: INT64[2]) -> (FLOAT[2, 2], FLOAT[2, 4]):

    output_1, output_2 = opset13.Split(input, split, axis=1)
    return output_1, output_2
