
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mvn(X: FLOAT[3, 3, 3, 1]) -> (FLOAT[3, 3, 3, 1]):

    Y = opset13.MeanVarianceNormalization(X)
    return Y
