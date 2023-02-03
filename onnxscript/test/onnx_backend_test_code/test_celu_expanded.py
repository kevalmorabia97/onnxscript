
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset12


@script()
def bck_test_celu_expanded(X: FLOAT[3, 3, 3, 1]) -> (FLOAT[3, 3, 3, 1]):

    Celu_test_celu_expanded_function_alpha = opset12.Constant(
        value=make_tensor("value", 1, dims=[1], vals=[2.0]))
    Celu_test_celu_expanded_function_X_alpha = opset12.Div(
        X, Celu_test_celu_expanded_function_alpha)
    Celu_test_celu_expanded_function_Elu_Result = opset12.Elu(
        Celu_test_celu_expanded_function_X_alpha, alpha=1.0)
    Y = opset12.Mul(Celu_test_celu_expanded_function_alpha,
                    Celu_test_celu_expanded_function_Elu_Result)
    return Y
