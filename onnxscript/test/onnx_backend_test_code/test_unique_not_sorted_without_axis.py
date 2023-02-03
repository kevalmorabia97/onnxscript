
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_unique_not_sorted_without_axis(X: FLOAT[6]) -> (FLOAT[4], INT64[4], INT64[6], INT64[4]):

    Y, indices, inverse_indices, counts = opset11.Unique(X, sorted=0)
    return Y, indices, inverse_indices, counts
