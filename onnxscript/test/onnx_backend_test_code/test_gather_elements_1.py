
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gather_elements_1(data: FLOAT[3, 3], indices: INT64[2, 3]) -> (FLOAT[2, 3]):

    y = opset13.GatherElements(data, indices, axis=0)
    return y
