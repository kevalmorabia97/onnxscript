
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_transpose_all_permutations_3(data: FLOAT[2, 3, 4]) -> (FLOAT[3, 4, 2]):

    transposed = opset13.Transpose(data, perm=[1, 2, 0])
    return transposed
