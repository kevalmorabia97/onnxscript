
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_top_k_smallest(x: FLOAT[3, 4], k: INT64[1]) -> (FLOAT[3, 3], INT64[3, 3]):

    values, indices = opset11.TopK(x, k, axis=1, largest=0, sorted=1)
    return values, indices
