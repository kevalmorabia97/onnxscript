
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.onnx_opset import opset9


@script()
def bck_test_tfidfvectorizer_tf_batch_onlybigrams_skip5(X: INT32[2, 6]) -> (FLOAT[2, 7]):

    Y = opset9.TfIdfVectorizer(X, max_gram_length=2, max_skip_count=5, min_gram_length=2, mode='TF', ngram_counts=[
                               0, 4], ngram_indexes=[0, 1, 2, 3, 4, 5, 6], pool_int64s=[2, 3, 5, 4, 5, 6, 7, 8, 6, 7])
    return Y
