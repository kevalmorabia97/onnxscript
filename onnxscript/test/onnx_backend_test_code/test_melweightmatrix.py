
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.onnx_opset import opset17


@script()
def bck_test_melweightmatrix(num_mel_bins: INT32, dft_length: INT32, sample_rate: INT32, lower_edge_hertz: FLOAT, upper_edge_hertz: FLOAT) -> (FLOAT[9, 8]):

    output = opset17.MelWeightMatrix(
        num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)
    return output
