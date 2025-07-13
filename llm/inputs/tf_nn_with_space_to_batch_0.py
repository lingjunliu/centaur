
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_with_space_to_batch_inputs():
    list_of_inputs = []

    def dummy_op(input, num_spatial_dims, padding):
        return input

    # Input 1
    input = np.random.rand(1, 4, 4, 1).astype(np.float32)
    dilation_rate = np.array([2, 2]).astype(np.int32)
    padding = "VALID"
    op = dummy_op
    filter_shape = np.array([3, 3]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.random.rand(1, 8, 8, 3).astype(np.float32)
    dilation_rate = np.array([4, 4]).astype(np.int32)
    padding = "SAME"
    op = dummy_op
    filter_shape = np.array([5, 5]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.random.rand(1, 16, 16, 1).astype(np.float32)
    dilation_rate = np.array([8, 8]).astype(np.int32)
    padding = "VALID"
    op = dummy_op
    filter_shape = np.array([1, 1]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.random.rand(1, 5, 5, 1).astype(np.float32)
    dilation_rate = np.array([1, 1]).astype(np.int32)
    padding = "SAME"
    op = dummy_op
    filter_shape = np.array([3, 3]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.random.rand(2, 6, 6, 3).astype(np.float32)
    dilation_rate = np.array([3, 3]).astype(np.int32)
    padding = "VALID"
    op = dummy_op
    filter_shape = np.array([1, 1]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.random.rand(1, 3, 3, 1).astype(np.float32)
    dilation_rate = np.array([2, 2]).astype(np.int32)
    padding = "SAME"
    op = dummy_op
    filter_shape = np.array([2, 2]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input = np.random.rand(2, 4, 5, 3, 1).astype(np.float32)
    dilation_rate = np.array([2, 2, 2]).astype(np.int32)
    padding = "VALID"
    op = dummy_op
    filter_shape = np.array([1, 1, 1]).astype(np.int32)
    spatial_dims = [1, 2, 3]
    data_format = "NDHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.random.rand(1, 7, 8, 9, 3).astype(np.float32)
    dilation_rate = np.array([2, 3, 4]).astype(np.int32)
    padding = "SAME"
    op = dummy_op
    filter_shape = np.array([3, 3, 3]).astype(np.int32)
    spatial_dims = [1, 2, 3]
    data_format = "NDHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    dilation_rate = np.array([1, 1]).astype(np.int32)
    padding = "VALID"
    op = dummy_op
    filter_shape = np.array([1, 1]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.random.rand(4, 16, 16, 8).astype(np.float32)
    dilation_rate = np.array([4, 4]).astype(np.int32)
    padding = "SAME"
    op = dummy_op
    filter_shape = np.array([7, 7]).astype(np.int32)
    spatial_dims = [1, 2]
    data_format = "NHWC"

    input_dict = {
        "input": input,
        "dilation_rate": dilation_rate,
        "padding": padding,
        "op": op,
        "filter_shape": filter_shape,
        "spatial_dims": spatial_dims,
        "data_format": data_format
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
def check_valid(api, input_list, lib="tf", suffix=0):
    from utils.new_api_utils import run_api, get_signature
    from generator.input_generators import get_abstract_input
    for input_dict in input_list:
        try:
            _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
            output = run_api(api, input_dict, cpu=True, lib=lib)
        except Exception as e:
            print(f"Error for input: {input_dict}. Error: {e}")
            raise e

generated_inputs["tf.nn.with_space_to_batch"] = tf_nn_with_space_to_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.with_space_to_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.with_space_to_batch'.")

check_valid('tf.nn.with_space_to_batch', generated_inputs['tf.nn.with_space_to_batch'], lib="tf", suffix=0)
