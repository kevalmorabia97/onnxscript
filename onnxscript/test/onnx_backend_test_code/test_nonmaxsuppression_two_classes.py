
import numpy
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxscript import script, external_tensor
from onnxscript.values import Opset

from onnxscript.onnx_types import FLOAT, INT64
from onnxscript.onnx_opset import opset11


@script()
def bck_test_nonmaxsuppression_two_classes(boxes: FLOAT[1, 6, 4], scores: FLOAT[1, 2, 6], max_output_boxes_per_class: INT64[1], iou_threshold: FLOAT[1], score_threshold: FLOAT[1]) -> (INT64[4, 3]):

    selected_indices = opset11.NonMaxSuppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    return selected_indices
