
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset16


@script()
def bck_test_scatternd_add(data: FLOAT[4, 4, 4], indices: INT64[2, 1], updates: FLOAT[2, 4, 4]) -> (FLOAT[4, 4, 4]):

    y = opset16.ScatterND(data, indices, updates, reduction='add')
    return y
