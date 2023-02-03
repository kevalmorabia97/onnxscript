
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset16


@script()
def bck_test_scatter_elements_without_axis(data: FLOAT[3, 3], indices: INT64[2, 3], updates: FLOAT[2, 3]) -> (FLOAT[3, 3]):

    y = opset16.ScatterElements(data, indices, updates)
    return y
