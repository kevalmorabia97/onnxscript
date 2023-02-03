
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import INT32, INT64
from onnxscript.onnx_opset import opset13


@script()
def bck_test_gathernd_example_int32_batch_dim1(data: INT32[2, 2, 2], indices: INT64[2, 1]) -> (INT32[2, 2]):

    output = opset13.GatherND(data, indices, batch_dims=1)
    return output
