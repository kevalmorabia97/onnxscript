
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_unique_sorted_with_axis_3d(X: FLOAT[2, 4, 2]) -> (FLOAT[2, 3, 2], INT64[3], INT64[4], INT64[3]):

    Y, indices, inverse_indices, counts = opset11.Unique(X, axis=1, sorted=1)
    return Y, indices, inverse_indices, counts
