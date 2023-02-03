
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT64
from onnxscript.onnx_opset import opset14


@script()
def bck_test_tril(x: INT64[4, 5]) -> (INT64[4, 5]):

    y = opset14.Trilu(x, upper=0)
    return y
