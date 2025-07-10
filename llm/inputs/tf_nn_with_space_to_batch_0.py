
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_with_space_to_batch_inputs():
    list_of_inputs = []

    def dummy_op(input_tensor, num_spatial_dims, padding):
        return input_tensor

    # Input 1
    input_tensor = tf.constant(np.random.rand(1, 4, 4, 1), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([2, 2], dtype=tf.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([3, 3], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(np.random.rand(1, 8, 8, 3), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([3, 3], dtype=tf.int32)
    padding_str = "SAME"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([5, 5], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(np.random.rand(1, 16, 16, 1), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([4, 4], dtype=tf.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([1, 1], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant(np.random.rand(1, 5, 5, 3), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([1, 1], dtype=tf.int32)
    padding_str = "SAME"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([3, 3], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant(np.random.rand(1, 3, 3, 1), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([2, 2], dtype=tf.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([1, 1], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant(np.random.rand(1, 4, 4, 1), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([2, 2], dtype=tf.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([3, 3], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant(np.random.rand(1, 8, 8, 3), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([3, 3], dtype=tf.int32)
    padding_str = "SAME"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([5, 5], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant(np.random.rand(1, 16, 16, 1), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([4, 4], dtype=tf.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([1, 1], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant(np.random.rand(1, 5, 5, 3), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([1, 1], dtype=tf.int32)
    padding_str = "SAME"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([3, 3], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
        "spatial_dims": spatial_dims_list,
        "data_format": data_format_str
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant(np.random.rand(1, 3, 3, 1), dtype=tf.float32)
    dilation_rate_tensor = tf.constant([2, 2], dtype=tf.int32)
    padding_str = "VALID"
    op_func = dummy_op
    filter_shape_tensor = tf.constant([1, 1], dtype=tf.int32)
    spatial_dims_list = [1, 2]
    data_format_str = "NHWC"

    input_dict = {
        "input": input_tensor.numpy(),
        "dilation_rate": dilation_rate_tensor.numpy(),
        "padding": padding_str,
        "op": [op_func],
        "filter_shape": filter_shape_tensor.numpy(),
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
    
    print("Valid")

if 'tf.nn.with_space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.with_space_to_batch'.")

check_valid('tf.nn.with_space_to_batch', generated_inputs['tf.nn.with_space_to_batch'], lib="tf", suffix=0)
