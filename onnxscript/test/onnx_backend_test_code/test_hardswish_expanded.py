
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset14


@script()
def bck_test_hardswish_expanded(x: FLOAT[3, 4, 5]) -> (FLOAT[3, 4, 5]):

    HardSwish_test_hardswish_expanded_function_HS_X = opset14.HardSigmoid(
        x, alpha=0.1666666716337204, beta=0.5)
    y = opset14.Mul(x, HardSwish_test_hardswish_expanded_function_HS_X)
    return y
