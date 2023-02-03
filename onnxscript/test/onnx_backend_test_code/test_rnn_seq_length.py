
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset14


@script()
def bck_test_rnn_seq_length(X: FLOAT[2, 3, 3], W: FLOAT[1, 5, 3], R: FLOAT[1, 5, 5], B: FLOAT[1, 10]) -> (FLOAT[1, 3, 5]):

    _0, Y_h = opset14.RNN(X, W, R, B, hidden_size=5)
    return Y_h
