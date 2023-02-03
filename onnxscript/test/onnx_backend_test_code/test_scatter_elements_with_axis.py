
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset16


@script()
def bck_test_scatter_elements_with_axis(data: FLOAT[1, 5], indices: INT64[1, 2], updates: FLOAT[1, 2]) -> (FLOAT[1, 5]):

    y = opset16.ScatterElements(data, indices, updates, axis=1)
    return y
