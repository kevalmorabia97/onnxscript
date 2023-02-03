
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset9


@script()
def bck_test_constantofshape_float_ones(x: INT64[3]) -> (FLOAT[4, 3, 2]):

    y = opset9.ConstantOfShape(x, value=make_tensor(
        "value", 1, dims=[1], vals=[1.0]))
    return y
