
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_edge_pad(x: INT32[1, 3, 4, 5], pads: INT64[8]) -> (INT32[1, 3, 6, 7]):

    y = opset13.Pad(x, pads, mode='edge')
    return y
