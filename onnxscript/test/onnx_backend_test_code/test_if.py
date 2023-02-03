
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_if(cond: BOOL) -> (FLOAT[5]):

    if cond:
        then_out = opset11.Constant(value=make_tensor(
            "value", 1, dims=[5], vals=[1.0, 2.0, 3.0, 4.0, 5.0]))
        res = then_out
    else:
        else_out = opset11.Constant(value=make_tensor(
            "value", 1, dims=[5], vals=[5.0, 4.0, 3.0, 2.0, 1.0]))
        res = else_out
    return res
