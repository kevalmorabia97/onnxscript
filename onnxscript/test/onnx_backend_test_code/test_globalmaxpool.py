
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset1


@script()
def bck_test_globalmaxpool(x: FLOAT[1, 3, 5, 5]) -> (FLOAT[1, 3, 1, 1]):

    y = opset1.GlobalMaxPool(x)
    return y
