
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSegmentSumGrad_inputs():
    list_of_inputs = []

    # Input 1: Simple case with float32
    grad = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    output_dim0 = np.array(3).astype(np.int32)
    name = "sparse_segment_sum_grad_1"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float64) and different segment_ids
    grad = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
    indices = np.array([0, 1, 2, 3]).astype(np.int64)
    segment_ids = np.array([0, 1, 0, 1]).astype(np.int64)
    output_dim0 = np.array(4).astype(np.int32)
    name = "sparse_segment_sum_grad_2"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: More complex segment_ids
    grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2]).astype(np.int32)
    output_dim0 = np.array(5).astype(np.int32)
    name = "sparse_segment_sum_grad_3"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different indices
    grad = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    indices = np.array([2, 0, 1]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    output_dim0 = np.array(3).astype(np.int32)
    name = "sparse_segment_sum_grad_4"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: One segment
    grad = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 0]).astype(np.int32)
    output_dim0 = np.array(3).astype(np.int32)
    name = "sparse_segment_sum_grad_5"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Float16 data type
    grad = np.array([1.0, 2.0, 3.0]).astype(np.float16)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    output_dim0 = np.array(3).astype(np.int32)
    name = "sparse_segment_sum_grad_6"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 data type, different values
    grad = np.array([0.5, 1.5, 2.5]).astype(np.float16)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    output_dim0 = np.array(3).astype(np.int32)
    name = "sparse_segment_sum_grad_7"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different output dim0
    grad = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    indices = np.array([0, 1, 2]).astype(np.int32)
    segment_ids = np.array([0, 0, 1]).astype(np.int32)
    output_dim0 = np.array(5).astype(np.int32)
    name = "sparse_segment_sum_grad_8"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More segment IDs
    grad = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5]).astype(np.int32)
    segment_ids = np.array([0, 0, 1, 1, 2, 2]).astype(np.int32)
    output_dim0 = np.array(6).astype(np.int32)
    name = "sparse_segment_sum_grad_9"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty grad array
    grad = np.array([]).astype(np.float32)
    indices = np.array([]).astype(np.int32)
    segment_ids = np.array([]).astype(np.int32)
    output_dim0 = np.array(0).astype(np.int32)
    name = "sparse_segment_sum_grad_10"
    input_dict = {"grad": grad, "indices": indices, "segment_ids": segment_ids, "output_dim0": output_dim0, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSegmentSumGrad"] = tf_raw_ops_SparseSegmentSumGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSegmentSumGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSegmentSumGrad'.")

check_valid('tf.raw_ops.SparseSegmentSumGrad', generated_inputs['tf.raw_ops.SparseSegmentSumGrad'], lib="tf", suffix=0)
