
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_dropout_default_mask(x: FLOAT[3, 4, 5]) -> (FLOAT[3, 4, 5], BOOL[3, 4, 5]):

    y, z = opset13.Dropout(x, seed=0)
    return y, z
