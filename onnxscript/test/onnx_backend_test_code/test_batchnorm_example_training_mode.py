
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15


@script()
def bck_test_batchnorm_example_training_mode(x: FLOAT[2, 3, 4, 5], s: FLOAT[3], bias: FLOAT[3], mean: FLOAT[3], var: FLOAT[3]) -> (FLOAT[2, 3, 4, 5], FLOAT[3], FLOAT[3]):

    y, output_mean, output_var = opset15.BatchNormalization(
        x, s, bias, mean, var, training_mode=1)
    return y, output_mean, output_var
