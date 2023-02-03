
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset6


@script()
def bck_test_instancenorm_epsilon(x: FLOAT[2, 3, 4, 5], s: FLOAT[3], bias: FLOAT[3]) -> (FLOAT[2, 3, 4, 5]):

    y = opset6.InstanceNormalization(x, s, bias, epsilon=0.009999999776482582)
    return y
