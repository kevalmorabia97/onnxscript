
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset16


@script()
def bck_test_gridsample_reflection_padding(X: FLOAT[1, 1, 3, 2], Grid: FLOAT[1, 2, 4, 2]) -> (FLOAT[1, 1, 2, 4]):

    Y = opset16.GridSample(X, Grid, padding_mode='reflection')
    return Y
