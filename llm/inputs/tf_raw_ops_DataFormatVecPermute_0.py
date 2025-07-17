
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DataFormatVecPermute_inputs():
    list_of_inputs = []

    # Input 1: Basic NHWC to NCHW, rank 1
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    src_format = "NHWC"
    dst_format = "NCHW"
    name = None
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NDHWC to NCDHW, rank 2
    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]], dtype=np.int64)
    src_format = "NDHWC"
    dst_format = "NCDHW"
    name = "my_permute"
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: NHWC to NCHW, rank 1, omitting non-spatial dimensions (size 2)
    x = np.array([1, 2], dtype=np.int32)
    src_format = "NHWC"
    dst_format = "NCHW"
    name = None
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: NCHW to NHWC, rank 1
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    src_format = "NCHW"
    dst_format = "NHWC"
    name = None
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: NCDHW to NDHWC, rank 2
    x = np.array([[1, 6], [5, 10], [2, 7], [3, 8], [4, 9]], dtype=np.int32)
    src_format = "NCDHW"
    dst_format = "NDHWC"
    name = "another_permute"
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: NHWC to NCHW, rank 2 (n, 2)
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    src_format = "NHWC"
    dst_format = "NCHW"
    name = None
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: NCHW to NHWC, rank 2 (n, 2)
    x = np.array([[1, 2], [4, 3], [2, 1], [3, 4]], dtype=np.int32)
    src_format = "NCHW"
    dst_format = "NHWC"
    name = None
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank 1, Custom format permutation
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    src_format = "ABCD"
    dst_format = "BADC"
    name = None
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Rank 2, Custom format permutation
    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9]], dtype=np.int32)
    src_format = "ABCD"
    dst_format = "CADB"
    name = "custom_permute"
    input_dict = {"x": x, "src_format": src_format, "dst_format": dst_format, "name": name}
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
