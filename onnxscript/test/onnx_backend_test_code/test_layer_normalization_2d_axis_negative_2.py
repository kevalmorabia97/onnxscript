
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset17


@script()
def bck_test_layer_normalization_2d_axis_negative_2(X: FLOAT[3, 4], W: FLOAT[3, 4], B: FLOAT[3, 4]) -> (FLOAT[3, 4], FLOAT[1, 1], FLOAT[1, 1]):

    Y, Mean, InvStdDev = opset17.LayerNormalization(X, W, B, axis=-2)
    return Y, Mean, InvStdDev
