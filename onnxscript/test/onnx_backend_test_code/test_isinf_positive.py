
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import BOOL, FLOAT
from onnxscript.onnx_opset import opset10


@script()
def bck_test_isinf_positive(x: FLOAT[6]) -> (BOOL[6]):

    y = opset10.IsInf(x, detect_negative=0)
    return y
