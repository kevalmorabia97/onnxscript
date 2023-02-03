
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset14


@script()
def bck_test_reshape_zero_dim(data: FLOAT[2, 3, 4], shape: INT64[4]) -> (FLOAT[2, 3, 4, 1]):

    reshaped = opset14.Reshape(data, shape)
    return reshaped
