
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_unique_sorted_with_negative_axis(X: FLOAT[3, 3]) -> (FLOAT[3, 2], INT64[2], INT64[3], INT64[2]):

    Y, indices, inverse_indices, counts = opset11.Unique(X, axis=-1, sorted=1)
    return Y, indices, inverse_indices, counts
