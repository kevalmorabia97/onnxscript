
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gather_2d_indices(data: FLOAT[3, 3], indices: INT64[1, 2]) -> (FLOAT[3, 1, 2]):

    y = opset13.Gather(data, indices, axis=1)
    return y
