
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import DOUBLE, INT32
from onnxscript.onnx_opset import opset9


@script()
def bck_test_eyelike_with_dtype(x: INT32[3, 4]) -> (DOUBLE[3, 4]):

    y = opset9.EyeLike(x, dtype=11)
    return y
