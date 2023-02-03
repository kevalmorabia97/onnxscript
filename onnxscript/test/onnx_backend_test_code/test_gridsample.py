
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_gridsample(X: FLOAT[1, 1, 4, 4], Grid: FLOAT[1, 6, 6, 2]) -> (FLOAT[1, 1, 6, 6]):

    Y = opset16.GridSample(X, Grid, align_corners=0,
                           mode='bilinear', padding_mode='zeros')
    return Y
