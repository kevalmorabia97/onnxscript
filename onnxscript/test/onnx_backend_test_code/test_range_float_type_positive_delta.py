
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_range_float_type_positive_delta(start: FLOAT, limit: FLOAT, delta: FLOAT) -> (FLOAT[2]):

    output = opset11.Range(start, limit, delta)
    return output
