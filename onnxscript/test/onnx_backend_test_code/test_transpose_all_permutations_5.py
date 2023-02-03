
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_transpose_all_permutations_5(data: FLOAT[2, 3, 4]) -> (FLOAT[4, 3, 2]):

    transposed = opset13.Transpose(data, perm=[2, 1, 0])
    return transposed
