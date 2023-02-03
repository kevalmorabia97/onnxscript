
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset14


@script()
def bck_test_simple_rnn_defaults(X: FLOAT[1, 3, 2], W: FLOAT[1, 4, 2], R: FLOAT[1, 4, 4]) -> (FLOAT[1, 3, 4]):

    _0, Y_h = opset14.RNN(X, W, R, hidden_size=4)
    return Y_h
