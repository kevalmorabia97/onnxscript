
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_compress_default_axis(input: FLOAT[3, 2], condition: BOOL[5]) -> (FLOAT[2]):

    output = opset11.Compress(input, condition)
    return output
