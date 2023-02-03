
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT
from onnxscript.onnx_opset import opset13


@script()
def bck_test_mvn_expanded(X: FLOAT[3, 3, 3, 1]) -> (FLOAT[3, 3, 3, 1]):

    MeanVarianceNormalization_test_mvn_expanded_function_Exponent = opset13.Constant(
        value=make_tensor("value", 1, dims=[], vals=[2.0]))
    MeanVarianceNormalization_test_mvn_expanded_function_Epsilon = opset13.Constant(
        value=make_tensor("value", 1, dims=[], vals=[9.999999717180685e-10]))
    MeanVarianceNormalization_test_mvn_expanded_function_X_RM = opset13.ReduceMean(X, axes=[
                                                                                   0, 2, 3])
    MeanVarianceNormalization_test_mvn_expanded_function_EX_squared = opset13.Pow(
        MeanVarianceNormalization_test_mvn_expanded_function_X_RM, MeanVarianceNormalization_test_mvn_expanded_function_Exponent)
    MeanVarianceNormalization_test_mvn_expanded_function_X_squared = opset13.Pow(
        X, MeanVarianceNormalization_test_mvn_expanded_function_Exponent)
    MeanVarianceNormalization_test_mvn_expanded_function_E_Xsquared = opset13.ReduceMean(
        MeanVarianceNormalization_test_mvn_expanded_function_X_squared, axes=[0, 2, 3])
    MeanVarianceNormalization_test_mvn_expanded_function_Variance = opset13.Sub(
        MeanVarianceNormalization_test_mvn_expanded_function_E_Xsquared, MeanVarianceNormalization_test_mvn_expanded_function_EX_squared)
    MeanVarianceNormalization_test_mvn_expanded_function_STD = opset13.Sqrt(
        MeanVarianceNormalization_test_mvn_expanded_function_Variance)
    MeanVarianceNormalization_test_mvn_expanded_function_X_variance = opset13.Sub(
        X, MeanVarianceNormalization_test_mvn_expanded_function_X_RM)
    MeanVarianceNormalization_test_mvn_expanded_function_Processed_STD = opset13.Add(
        MeanVarianceNormalization_test_mvn_expanded_function_STD, MeanVarianceNormalization_test_mvn_expanded_function_Epsilon)
    Y = opset13.Div(MeanVarianceNormalization_test_mvn_expanded_function_X_variance,
                    MeanVarianceNormalization_test_mvn_expanded_function_Processed_STD)
    return Y
