
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset17


@script()
def bck_test_layer_normalization_4d_axis0(X: FLOAT[2, 3, 4, 5], W: FLOAT[2, 3, 4, 5], B: FLOAT[2, 3, 4, 5]) -> (FLOAT[2, 3, 4, 5], FLOAT[1, 1, 1, 1], FLOAT[1, 1, 1, 1]):

    Y, Mean, InvStdDev = opset17.LayerNormalization(X, W, B, axis=0)
    return Y, Mean, InvStdDev
