
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset7


@script()
def bck_test_tan_example(x: FLOAT[3]) -> (FLOAT[3]):

    y = opset7.Tan(x)
    return y
