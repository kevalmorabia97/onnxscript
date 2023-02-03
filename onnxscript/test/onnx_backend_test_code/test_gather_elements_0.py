
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gather_elements_0(data: FLOAT[2, 2], indices: INT64[2, 2]) -> (FLOAT[2, 2]):

    y = opset13.GatherElements(data, indices, axis=1)
    return y
