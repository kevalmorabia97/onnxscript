
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_spacetodepth(x: FLOAT[2, 2, 6, 6]) -> (FLOAT[2, 8, 3, 3]):

    y = opset13.SpaceToDepth(x, blocksize=2)
    return y
