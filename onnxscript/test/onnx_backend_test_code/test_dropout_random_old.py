
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_dropout_random_old(x: FLOAT[3, 4, 5]) -> (FLOAT[3, 4, 5]):

    y = opset11.Dropout(x, ratio=0.20000000298023224)
    return y
