
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_depthtospace_crd_mode_example(x: FLOAT[1, 8, 2, 3]) -> (FLOAT[1, 2, 4, 6]):

    y = opset13.DepthToSpace(x, blocksize=2, mode='CRD')
    return y
