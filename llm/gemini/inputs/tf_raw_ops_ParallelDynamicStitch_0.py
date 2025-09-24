
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_parallel_dynamic_stitch_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.ParallelDynamicStitch.
    The 'indices' and 'data' arguments, which are 'tensor_list' types, are
    provided as single stacked NumPy arrays to accommodate a testing framework
    that expects a .shape attribute and does not correctly handle Python lists
    for this type. This requires all tensors in a list to be homogeneous in shape.
    """
    list_of_inputs = []

    # Input 1: Basic vector indices
    input_dict_1 = {
        'indices': np.array([[1, 3], [0, 2]], dtype=np.int32),
        'data': np.array([[10, 30], [5, 25]], dtype=np.int64),
        'name': 'vector_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Scalar indices (stacked into a 1D array)
    input_dict_2 = {
        'indices': np.array([0, 2], dtype=np.int32),
        'data': np.array([[10.0, 11.0], [30.0, 31.0]], dtype=np.float32),
        'name': 'scalar_indices_stacked'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Higher rank data slices
    input_dict_3 = {
        'indices': np.array([[0, 2], [1, 3]], dtype=np.int32),
        'data': np.stack([
            np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            np.arange(12, 24, dtype=np.float32).reshape(2, 2, 3)
        ]),
        'name': 'high_rank_data'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Single partition (stacked into a tensor with a leading dim of 1)
    input_dict_4 = {
        'indices': np.array([[2, 0, 1]], dtype=np.int32),
        'data': np.array([[[10, 11], [20, 21], [30, 31]]], dtype=np.int32),
        'name': 'single_partition_stacked'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Empty tensors (stacked)
    input_dict_5 = {
        'indices': np.empty((2, 0), dtype=np.int32),
        'data': np.empty((2, 0, 5), dtype=np.float32),
        'name': 'empty_tensors_stacked'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Boolean data type
    input_dict_6 = {
        'indices': np.array([[1, 0], [3, 2]], dtype=np.int32),
        'data': np.array([[True, False], [False, True]], dtype=bool),
        'name': 'boolean_data'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: uint8 data type
    input_dict_7 = {
        'indices': np.array([[0], [1]], dtype=np.int32),
        'data': np.array([[[10]], [[20]]], dtype=np.uint8),
        'name': 'uint8_data'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: High rank indices
    input_dict_8 = {
        'indices': np.array([[[0], [2]], [[1], [3]]], dtype=np.int32),
        'data': np.array([[[[0.1]], [[2.1]]], [[[1.1]], [[3.1]]]], dtype=np.float32),
        'name': 'high_rank_indices'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Multiple partitions
    input_dict_9 = {
        'indices': np.arange(4, dtype=np.int32).reshape(4, 1),
        'data': np.stack([np.array([[i, i+1]], dtype=np.int32) for i in range(4)]),
        'name': 'many_partitions'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: name=None
    input_dict_10 = {
        'indices': np.array([[1], [0]], dtype=np.int32),
        'data': np.array([[1.0], [2.0]], dtype=np.float64),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.ParallelDynamicStitch"] = tf_raw_ops_parallel_dynamic_stitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ParallelDynamicStitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ParallelDynamicStitch'.")

check_valid('tf.raw_ops.ParallelDynamicStitch', generated_inputs['tf.raw_ops.ParallelDynamicStitch'], lib="tf", suffix=0)
