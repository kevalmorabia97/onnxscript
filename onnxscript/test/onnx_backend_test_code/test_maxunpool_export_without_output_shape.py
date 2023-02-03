
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_maxunpool_export_without_output_shape(xT: FLOAT[1, 1, 2, 2], xI: INT64[1, 1, 2, 2]) -> (FLOAT[1, 1, 4, 4]):

    y = opset11.MaxUnpool(xT, xI, kernel_shape=[2, 2], strides=[2, 2])
    return y
