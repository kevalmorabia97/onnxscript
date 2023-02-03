
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_resize_upsample_scales_linear_align_corners(X: FLOAT[1, 1, 2, 2], scales: FLOAT[4]) -> (FLOAT[1, 1, 4, 4]):

    Y = opset13.Resize(
        X, None, scales, coordinate_transformation_mode='align_corners', mode='linear')
    return Y
