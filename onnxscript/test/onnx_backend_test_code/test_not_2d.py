
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL
from onnxscript.onnx_opset import opset1


@script()
def bck_test_not_2d(x: BOOL[3, 4]) -> (BOOL[3, 4]):

    r_not = opset1.Not(x)
    return r_not
