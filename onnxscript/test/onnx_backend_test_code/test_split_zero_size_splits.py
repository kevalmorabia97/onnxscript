
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_split_zero_size_splits(input: FLOAT[0], split: INT64[3]) -> (FLOAT[0], FLOAT[0], FLOAT[0]):

    output_1, output_2, output_3 = opset13.Split(input, split)
    return output_1, output_2, output_3
