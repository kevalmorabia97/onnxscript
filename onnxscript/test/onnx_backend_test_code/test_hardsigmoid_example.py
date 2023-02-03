
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset6


@script()
def bck_test_hardsigmoid_example(x: FLOAT[3]) -> (FLOAT[3]):

    y = opset6.HardSigmoid(x, alpha=0.5, beta=0.6000000238418579)
    return y
