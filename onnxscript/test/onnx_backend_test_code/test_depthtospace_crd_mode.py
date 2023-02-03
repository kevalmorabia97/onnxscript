
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset11


@script()
def bck_test_depthtospace_crd_mode(x: FLOAT[2, 8, 3, 3]) -> (FLOAT[2, 2, 6, 6]):

    y = opset11.DepthToSpace(x, blocksize=2, mode='CRD')
    return y
