
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn(X: FLOAT[1, 1, 4, 4], sizes: INT64[4]) -> (FLOAT[1, 1, 3, 2]):

    roi = opset11.Constant(value=make_tensor("value", 1, dims=[0], vals=[]))
    scales = opset11.Constant(value=make_tensor("value", 1, dims=[0], vals=[]))
    Y = opset11.Resize(X, roi, scales, sizes,
                       coordinate_transformation_mode='tf_half_pixel_for_nn', mode='nearest')
    return Y
