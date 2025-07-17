
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_with_space_to_batch_inputs():
    list_of_inputs = []

    def op1(input, num_spatial_dims, padding):
        return input * 2

    def op2(input, num_spatial_dims, padding):
        return tf.nn.relu(input)

    def op3(input, num_spatial_dims, padding):
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding=padding)

    def op4(input, num_spatial_dims, padding):
      return tf.nn.max_pool3d(input, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding=padding)

    def op5(input, num_spatial_dims, padding):
      return tf.nn.avg_pool3d(input, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding=padding)

    # Input 1
    input_tensor = np.random.rand(1, 4, 4, 3).astype(np.float32)
    dilation_rate = np.array([2, 2]).astype(np.int32)
    padding = "VALID"
    op = op1
    filter_shape = np.array([3, 3]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 8, 8, 1).astype(np.float32)
    dilation_rate = np.array([3, 3]).astype(np.int32)
    padding = "SAME"
    op = op2
    filter_shape = np.array([5, 5]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(2, 16, 16, 3).astype(np.float32)
    dilation_rate = np.array([1, 1]).astype(np.int32)
    padding = "VALID"
    op = op3
    filter_shape = np.array([2, 2]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    dilation_rate = np.array([2, 2]).astype(np.int32)
    padding = "SAME"
    op = op1
    filter_shape = np.array([3, 3]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 3, 3, 3).astype(np.float32)
    dilation_rate = np.array([1, 1]).astype(np.int32)
    padding = "VALID"
    op = op2
    filter_shape = np.array([1, 1]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 4, 4, 3).astype(np.float32)
    dilation_rate = np.array([2, 2]).astype(np.int32)
    padding = "VALID"
    op = op1
    filter_shape = np.array([3, 3]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NCHW"
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.random.rand(1, 5, 5, 3).astype(np.float32)
    dilation_rate = np.array([2, 2]).astype(np.int32)
    padding = "SAME"
    op = op1
    filter_shape = np.array([3, 3]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NCHW"
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))


    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 3, 3, 3).astype(np.float32)
    dilation_rate = np.array([1, 1]).astype(np.int32)
    padding = "VALID"
    op = op2
    filter_shape = np.array([1, 1]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NCHW"
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 4, 4, 4, 3).astype(np.float32)
    dilation_rate = np.array([2, 2, 2]).astype(np.int32)
    padding = "VALID"

    op = op4
    filter_shape = np.array([3, 3, 3]).astype(np.int32)
    spatial_dims = [1, 2, 3]
    data_format = "NDHWC"

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 4, 4, 4, 3).astype(np.float32)
    dilation_rate = np.array([2, 2, 2]).astype(np.int32)
    padding = "SAME"

    op = op5
    filter_shape = np.array([3, 3, 3]).astype(np.int32)
    spatial_dims = [1, 2, 3]
    data_format = "NCDHW"
    input_tensor = np.transpose(input_tensor, (0, 4, 1, 2, 3))

    input_dict = {
        "input": input_tensor,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.nn.with_space_to_batch"] = tf_nn_with_space_to_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.with_space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.with_space_to_batch'.")

check_valid('tf.nn.with_space_to_batch', generated_inputs['tf.nn.with_space_to_batch'], lib="tf", suffix=0)
