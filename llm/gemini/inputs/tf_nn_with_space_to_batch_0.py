
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_nn_with_space_to_batch_inputs():
    list_of_inputs = []

    # Define a dummy op for testing
    def dummy_op(input, num_spatial_dims, padding):
        return tf.identity(input)

    # Input 1: VALID padding, simple case
    input_np = np.random.rand(1, 4, 4, 3).astype(np.float32)
    dilation_rate_np = np.array([2, 2]).astype(np.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_np = np.array([3, 3]).astype(np.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_np,
        "dilation_rate": dilation_rate_np,
        "padding": padding_str,
        "op": op_func,
        "filter_shape": filter_shape_np,
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: SAME padding
    input_np = np.random.rand(1, 8, 8, 3).astype(np.float32)
    dilation_rate_np = np.array([3, 3]).astype(np.int32)
    padding_str = "SAME"
    op_func = dummy_op
    filter_shape_np = np.array([5, 5]).astype(np.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_np,
        "dilation_rate": dilation_rate_np,
        "padding": padding_str,
        "op": op_func,
        "filter_shape": filter_shape_np,
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data format
    input_np = np.random.rand(1, 3, 4, 4).astype(np.float32)
    dilation_rate_np = np.array([2, 2]).astype(np.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_np = np.array([3, 3]).astype(np.int32)
    spatial_dims_list = [2, 3]
    data_format_str = "NCHW"

    input_dict = {
        "input": input_np,
        "dilation_rate": dilation_rate_np,
        "padding": padding_str,
        "op": op_func,
        "filter_shape": filter_shape_np,
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D input
    input_np = np.random.rand(1, 4, 4, 4, 3).astype(np.float32)
    dilation_rate_np = np.array([2, 2, 2]).astype(np.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_np = np.array([3, 3, 3]).astype(np.int32)
    spatial_dims_list = [1, 2, 3]
    data_format_str = "NDHWC"

    input_dict = {
        "input": input_np,
        "dilation_rate": dilation_rate_np,
        "padding": padding_str,
        "op": op_func,
        "filter_shape": filter_shape_np,
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Spatial dims not starting from 1
    input_np = np.random.rand(1, 2, 4, 4, 3).astype(np.float32)
    dilation_rate_np = np.array([2, 2]).astype(np.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_np = np.array([3, 3]).astype(np.int32)
    spatial_dims_list = [2, 3]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_np,
        "dilation_rate": dilation_rate_np,
        "padding": padding_str,
        "op": op_func,
        "filter_shape": filter_shape_np,
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Uniform dilation rate of 1 (should be equivalent to original op)
    input_np = np.random.rand(1, 4, 4, 3).astype(np.float32)
    dilation_rate_np = np.array([1, 1]).astype(np.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_np = np.array([3, 3]).astype(np.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_np,
        "dilation_rate": dilation_rate_np,
        "padding": padding_str,
        "op": op_func,
        "filter_shape": filter_shape_np,
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.with_space_to_batch"] = tf_nn_with_space_to_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.with_space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.with_space_to_batch'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.with_space_to_batch', generated_inputs['tf.nn.with_space_to_batch'], lib="tf", suffix=0)
