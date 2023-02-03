
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset6


@script()
def bck_test_instancenorm_example(x: FLOAT[1, 2, 1, 3], s: FLOAT[2], bias: FLOAT[2]) -> (FLOAT[1, 2, 1, 3]):

    y = opset6.InstanceNormalization(x, s, bias)
    return y
