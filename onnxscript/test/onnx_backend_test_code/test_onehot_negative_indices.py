
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_onehot_negative_indices(indices: INT64[3], depth: FLOAT, values: FLOAT[2]) -> (FLOAT[3, 10]):

    y = opset11.OneHot(indices, depth, values, axis=1)
    return y
