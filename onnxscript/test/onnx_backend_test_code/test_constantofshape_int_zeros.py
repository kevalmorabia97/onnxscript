
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32, INT64
from onnxscript.onnx_opset import opset9


@script()
def bck_test_constantofshape_int_zeros(x: INT64[2]) -> (INT32[10, 6]):

    y = opset9.ConstantOfShape(
        x, value=make_tensor("value", 6, dims=[1], vals=[0]))
    return y
