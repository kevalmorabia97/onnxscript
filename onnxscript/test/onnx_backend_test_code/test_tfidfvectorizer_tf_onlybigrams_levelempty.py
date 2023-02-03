
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT32
from onnxscript.onnx_opset import opset9


@script()
def bck_test_tfidfvectorizer_tf_onlybigrams_levelempty(X: INT32[12]) -> (FLOAT[3]):

    Y = opset9.TfIdfVectorizer(X, max_gram_length=2, max_skip_count=0, min_gram_length=2, mode='TF', ngram_counts=[
                               0, 0], ngram_indexes=[0, 1, 2], pool_int64s=[5, 6, 7, 8, 6, 7])
    return Y
