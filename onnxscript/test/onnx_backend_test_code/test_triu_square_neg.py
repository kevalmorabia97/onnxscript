
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT64
from onnxscript.onnx_opset import opset14


@script()
def bck_test_triu_square_neg(x: INT64[2, 3, 3], k: INT64) -> (INT64[2, 3, 3]):

    y = opset14.Trilu(x, k)
    return y
