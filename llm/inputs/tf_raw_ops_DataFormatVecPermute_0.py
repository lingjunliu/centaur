
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatVecPermute_inputs():
    list_of_inputs = []

    # Input 1: Vector, default formats
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "src_format": "NHWC", "dst_format": "NCHW", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector, different formats
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "src_format": "NDHWC", "dst_format": "NCDHW", "name": "permute_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, default formats
    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9]], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "src_format": "NHWC", "dst_format": "NCHW", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor, different formats
    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "src_format": "NDHWC", "dst_format": "NCDHW", "name": "permute_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Vector, omitting non-spatial dimensions (n-2)
    x = np.array([1, 2], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "src_format": "NHWC", "dst_format": "NCHW", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor, omitting non-spatial dimensions (n-2, 2)
    x = np.array([[1, 6], [2, 7]], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "src_format": "NHWC", "dst_format": "NCHW", "name": "permute_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different format lengths, vector
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "src_format": "HWCN", "dst_format": "CHWN", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different format lengths, 2D tensor
    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9]], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "src_format": "HWCN", "dst_format": "CHWN", "name": "permute_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 2D tensor n-2 dimension
    x = np.array([[1, 6], [2, 7], [3, 8]], dtype=np.int32)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "src_format": "DHWCN", "dst_format": "CDHWN", "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Vector n-2 dimension
    x = np.array([1, 2, 3], dtype=np.int64)
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int64), "src_format": "DHWCN", "dst_format": "CDHWN", "name": "permute_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DataFormatVecPermute"] = tf_raw_ops_DataFormatVecPermute_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DataFormatVecPermute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DataFormatVecPermute'.")

check_valid('tf.raw_ops.DataFormatVecPermute', generated_inputs['tf.raw_ops.DataFormatVecPermute'], lib="tf", suffix=0)
