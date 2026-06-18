
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatVecPermute_inputs():
    list_of_inputs = []

    # Case 1: n=4 vector, size n
    input_dict = {
        'src_format': 'NHWC',
        'dst_format': 'NCHW',
        'name': 'permute_1',
        'x': np.array([1, 2, 3, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: n=4 vector, size n-2
    input_dict = {
        'src_format': 'NHWC',
        'dst_format': 'NCHW',
        'name': 'permute_2',
        'x': np.array([10, 20], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: n=5 tensor, shape (n, 2)
    input_dict = {
        'src_format': 'NDHWC',
        'dst_format': 'NCDHW',
        'name': 'permute_3',
        'x': np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: n=5 tensor, shape (n-2, 2)
    input_dict = {
        'src_format': 'NDHWC',
        'dst_format': 'NCDHW',
        'name': 'permute_4',
        'x': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: n=4 vector, size n (with negative numbers)
    input_dict = {
        'src_format': 'NCHW',
        'dst_format': 'NHWC',
        'name': 'permute_5',
        'x': np.array([-1, -2, -3, -4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: n=4 vector, size n-2
    input_dict = {
        'src_format': 'NCHW',
        'dst_format': 'NHWC',
        'name': 'permute_6',
        'x': np.array([-10, -20], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: n=5 tensor, shape (n, 2) with negative numbers
    input_dict = {
        'src_format': 'NCDHW',
        'dst_format': 'NDHWC',
        'name': 'permute_7',
        'x': np.array([[-1, 1], [-2, 2], [-3, 3], [-4, 4], [-5, 5]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: n=5 tensor, shape (n-2, 2)
    input_dict = {
        'src_format': 'NCDHW',
        'dst_format': 'NDHWC',
        'name': 'permute_8',
        'x': np.array([[-10, 10], [-20, 20], [-30, 30]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: n=4 tensor, shape (n, 2)
    input_dict = {
        'src_format': 'NHWC',
        'dst_format': 'NCHW',
        'name': 'permute_9',
        'x': np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: n=5 vector, size n
    input_dict = {
        'src_format': 'NDHWC',
        'dst_format': 'NCDHW',
        'name': 'permute_10',
        'x': np.array([100, 200, 300, 400, 500], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DataFormatVecPermute"] = tf_raw_ops_DataFormatVecPermute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DataFormatVecPermute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DataFormatVecPermute'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DataFormatVecPermute', generated_inputs['tf.raw_ops.DataFormatVecPermute'], lib="tf", suffix=0)
