
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset8


@script()
def bck_test_convtranspose_with_kernel(x: FLOAT[1, 1, 3, 3], w: FLOAT[1, 2, 3, 3]) -> (FLOAT[1, 2, 10, 8]):

    y = opset8.ConvTranspose(x, w, kernel_shape=[3, 3], output_padding=[
                             1, 1], output_shape=[10, 8], strides=[3, 2])
    return y
