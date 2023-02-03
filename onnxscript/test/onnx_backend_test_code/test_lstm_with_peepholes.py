
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.onnx_opset import opset14


@script()
def bck_test_lstm_with_peepholes(X: FLOAT[1, 2, 4], W: FLOAT[1, 12, 4], R: FLOAT[1, 12, 3], B: FLOAT[1, 24], sequence_lens: INT32[2], initial_h: FLOAT[1, 2, 3], initial_c: FLOAT[1, 2, 3], P: FLOAT[1, 9]) -> (FLOAT[1, 2, 3]):

    _0, Y_h = opset14.LSTM(X, W, R, B, sequence_lens,
                           initial_h, initial_c, P, hidden_size=3)
    return Y_h
