
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset14


@script()
def bck_test_lstm_with_initial_bias(X: FLOAT[1, 3, 3], W: FLOAT[1, 16, 3], R: FLOAT[1, 16, 4], B: FLOAT[1, 32]) -> (FLOAT[1, 3, 4]):

    _0, Y_h = opset14.LSTM(X, W, R, B, hidden_size=4)
    return Y_h
