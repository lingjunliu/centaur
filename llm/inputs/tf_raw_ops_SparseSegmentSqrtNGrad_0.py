
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSegmentSqrtNGrad_inputs():
    list_of_inputs = []

    # Input 1
    grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    output_dim0 = np.array(4, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 2, 5], dtype=np.int64)
    segment_ids = np.array([0, 0, 1], dtype=np.int64)
    output_dim0 = np.array(6, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    output_dim0 = np.array(3, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([3, 1, 0, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    output_dim0 = np.array(4, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    grad = np.array([1.0, 2.0], dtype=np.float16)
    indices = np.array([0, 1], dtype=np.int32)
    segment_ids = np.array([0, 0], dtype=np.int32)
    output_dim0 = np.array(2, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    output_dim0 = np.array(5, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: empty grad
    grad = np.array([], dtype=np.float32)
    indices = np.array([], dtype=np.int32)
    segment_ids = np.array([], dtype=np.int32)
    output_dim0 = np.array(0, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: more complex segment_ids
    grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    output_dim0 = np.array(6, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: different indices
    grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([1, 3, 5], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    output_dim0 = np.array(6, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: different output_dim0
    grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    output_dim0 = np.array(5, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    output_dim0 = np.array(4, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    segment_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    output_dim0 = np.array(4, dtype=np.int32)
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSegmentSqrtNGrad"] = tf_raw_ops_SparseSegmentSqrtNGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSegmentSqrtNGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSqrtNGrad'.")

check_valid('tf.raw_ops.SparseSegmentSqrtNGrad', generated_inputs['tf.raw_ops.SparseSegmentSqrtNGrad'], lib="tf", suffix=0)
