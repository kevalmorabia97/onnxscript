
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.onnx_opset import opset9


@script()
def bck_test_eyelike_populate_off_main_diagonal(x: INT32[4, 5]) -> (FLOAT[4, 5]):

    y = opset9.EyeLike(x, dtype=1, k=1)
    return y
