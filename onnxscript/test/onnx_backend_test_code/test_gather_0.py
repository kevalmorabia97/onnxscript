
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gather_0(data: FLOAT[5, 4, 3, 2], indices: INT64[3]) -> (FLOAT[3, 4, 3, 2]):

    y = opset13.Gather(data, indices, axis=0)
    return y
