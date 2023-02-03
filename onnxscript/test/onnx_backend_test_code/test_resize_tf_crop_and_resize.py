
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_resize_tf_crop_and_resize(X: FLOAT[1, 1, 4, 4], roi: FLOAT[8], sizes: INT64[4]) -> (FLOAT[1, 1, 3, 3]):

    Y = opset13.Resize(X, roi, None, sizes, coordinate_transformation_mode='tf_crop_and_resize',
                       extrapolation_value=10.0, mode='linear')
    return Y
