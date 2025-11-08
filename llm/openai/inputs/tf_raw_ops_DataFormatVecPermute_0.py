
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_data_format_vec_permute_inputs():
    list_of_inputs = []

    x = np.array([1, 2, 3, 4], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_1_nhwc_to_nchw_vec4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10, 20], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_2_nhwc_to_nchw_vec2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 6], [2, 7], [3, 8], [4, 9]], dtype=np.int32)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_3_nhwc_to_nchw_mat4x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[5, -5], [6, -6]], dtype=np.int64)
    input_dict = {
        "src_format": "NHWC",
        "dst_format": "NCHW",
        "name": "case_4_nhwc_to_nchw_mat2x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1, 0, -3, 4], dtype=np.int64)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "case_5_nchw_to_nhwc_vec4",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([9, -8, 7, -6, 5], dtype=np.int32)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_6_ndhwc_to_ncdhw_vec5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([100000, -200000, 300000], dtype=np.int64)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_7_ndhwc_to_ncdhw_vec3",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[10, -10], [20, -20], [30, -30], [40, -40], [50, -50]], dtype=np.int64)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_8_ndhwc_to_ncdhw_mat5x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    input_dict = {
        "src_format": "NDHWC",
        "dst_format": "NCDHW",
        "name": "case_9_ndhwc_to_ncdhw_mat3x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    input_dict = {
        "src_format": "NCDHW",
        "dst_format": "NDHWC",
        "name": "case_10_ncdhw_to_ndhwc_vec5",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[7, 14], [8, 16], [9, 18]], dtype=np.int64)
    input_dict = {
        "src_format": "NCDHW",
        "dst_format": "NDHWC",
        "name": "case_11_ncdhw_to_ndhwc_mat3x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1000, 2000], [3000, 4000], [5000, 6000], [7000, 8000]], dtype=np.int32)
    input_dict = {
        "src_format": "NCHW",
        "dst_format": "NHWC",
        "name": "case_12_nchw_to_nhwc_mat4x2",
        "x": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DataFormatVecPermute"] = tf_raw_ops_data_format_vec_permute_inputs()

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
