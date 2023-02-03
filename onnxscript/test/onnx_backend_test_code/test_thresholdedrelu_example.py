
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset10


@script()
def bck_test_thresholdedrelu_example(x: FLOAT[5]) -> (FLOAT[5]):

    y = opset10.ThresholdedRelu(x, alpha=2.0)
    return y
