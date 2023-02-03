
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset9


@script()
def bck_test_atanh_example(x: FLOAT[3]) -> (FLOAT[3]):

    y = opset9.Atanh(x)
    return y
