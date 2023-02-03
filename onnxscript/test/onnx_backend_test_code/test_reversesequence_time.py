
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset10


@script()
def bck_test_reversesequence_time(x: FLOAT[4, 4], sequence_lens: INT64[4]) -> (FLOAT[4, 4]):

    y = opset10.ReverseSequence(x, sequence_lens, batch_axis=1, time_axis=0)
    return y
