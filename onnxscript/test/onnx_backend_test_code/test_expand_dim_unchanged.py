
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_expand_dim_unchanged(data: FLOAT[3, 1], new_shape: INT64[2]) -> (FLOAT[3, 4]):

    expanded = opset13.Expand(data, new_shape)
    return expanded
