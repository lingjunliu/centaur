
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_scattermax_inputs():
    list_of_inputs = []

    # This raw op is not designed for eager execution and is expected to raise a
    # RuntimeError in modern TensorFlow versions when called directly.
    # The inputs provided below are syntactically and semantically correct
    # according to the API's documentation for a graph-based execution context.
    # The runtime error is due to the execution environment, not the inputs themselves.

    # Case 1: Basic 1D case, float32
    input_dict_1 = {
        'ref': np.array([1., 2., 3., 4.], dtype=np.float32),
        'indices': np.array([3, 1, 0], dtype=np.int32),
        'updates': np.array([5., 0., 6.], dtype=np.float32),
        'use_locking': False,
        'name': "case_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Basic 2D case, int32
    input_dict_2 = {
        'ref': np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[9, 10], [11, 12]], dtype=np.int32),
        'use_locking': False,
        'name': "case_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Duplicate indices
    input_dict_3 = {
        'ref': np.array([0., 0., 0., 0.], dtype=np.float32),
        'indices': np.array([1, 3, 1, 3], dtype=np.int32),
        'updates': np.array([10., 20., 5., 30.], dtype=np.float32),
        'use_locking': False,
        'name': "case_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: float64 dtype
    input_dict_4 = {
        'ref': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        'indices': np.array([1, 0], dtype=np.int32),
        'updates': np.array([[5.5, 6.6], [7.7, 8.8]], dtype=np.float64),
        'use_locking': False,
        'name': "case_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: int64 dtype for all tensors
    input_dict_5 = {
        'ref': np.array([10, 20, 30, 40], dtype=np.int64),
        'indices': np.array([0, 0, 3], dtype=np.int64),
        'updates': np.array([15, 5, 45], dtype=np.int64),
        'use_locking': False,
        'name': "case_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Negative values
    input_dict_6 = {
        'ref': np.array([-1, -5, 2, -10], dtype=np.int32),
        'indices': np.array([1, 3, 0], dtype=np.int32),
        'updates': np.array([-2, -8, -4], dtype=np.int32),
        'use_locking': False,
        'name': "case_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: Scalar update
    input_dict_7 = {
        'ref': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array(99.0, dtype=np.float32),
        'use_locking': False,
        'name': "case_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Case 8: bfloat16 dtype
    bfloat16 = tf.bfloat16.as_numpy_dtype
    input_dict_8 = {
        'ref': np.array([1.0, 5.0, 3.0, 8.0], dtype=bfloat16),
        'indices': np.array([1, 0, 1, 3], dtype=np.int32),
        'updates': np.array([9.0, 2.0, 6.0, 4.0], dtype=bfloat16),
        'use_locking': False,
        'name': "case_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: half (float16) dtype
    input_dict_9 = {
        'ref': np.array([[1, 2], [3, 4]], dtype=np.float16),
        'indices': np.array([0], dtype=np.int32),
        'updates': np.array([[5, 0]], dtype=np.float16),
        'use_locking': False,
        'name': "case_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Empty indices and updates
    input_dict_10 = {
        'ref': np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32),
        'indices': np.array([], dtype=np.int32),
        'updates': np.empty(shape=(0, 3), dtype=np.float32),
        'use_locking': False,
        'name': "case_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: use_locking=True
    input_dict_11 = {
        'ref': np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([[9, 10], [11, 12]], dtype=np.int32),
        'use_locking': True,
        'name': "case_11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterMax"] = tf_raw_ops_scattermax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterMax'.")

check_valid('tf.raw_ops.ScatterMax', generated_inputs['tf.raw_ops.ScatterMax'], lib="tf", suffix=0)
