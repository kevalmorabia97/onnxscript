
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset15


@script()
def bck_test_bernoulli_seed_expanded(x: FLOAT[10]) -> (FLOAT[10]):

    Bernoulli_test_bernoulli_seed_expanded_function_X_random = opset15.RandomUniformLike(
        x, low=0.0, high=1.0, seed=0.0, dtype=1)
    Bernoulli_test_bernoulli_seed_expanded_function_X_greater = opset15.Greater(
        Bernoulli_test_bernoulli_seed_expanded_function_X_random, x)
    y = opset15.Cast(
        Bernoulli_test_bernoulli_seed_expanded_function_X_greater, to=1)
    return y
