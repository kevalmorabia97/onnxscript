
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15


@script()
def bck_test_batchnorm_example(x: FLOAT[2, 3, 4, 5], s: FLOAT[3], bias: FLOAT[3], mean: FLOAT[3], var: FLOAT[3]) -> (FLOAT[2, 3, 4, 5]):

    y = opset15.BatchNormalization(x, s, bias, mean, var)
    return y
