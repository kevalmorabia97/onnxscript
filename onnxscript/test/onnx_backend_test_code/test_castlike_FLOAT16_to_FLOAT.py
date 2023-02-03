
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, FLOAT16
from onnxscript.onnx_opset import opset15


@script()
def bck_test_castlike_FLOAT16_to_FLOAT(input: FLOAT16[3, 4], like: FLOAT[1]) -> (FLOAT[3, 4]):

    output = opset15.CastLike(input, like)
    return output
