
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool_with_argmax_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant(np.random.rand(1, 5, 5, 1).astype(np.float32))
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    output_dtype = tf.int64
    include_batch_in_index = False
    name = None

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(np.random.rand(1, 10, 10, 3).astype(np.float32))
    ksize = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    output_dtype = tf.int32
    include_batch_in_index = True
    name = "max_pool"

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(np.random.rand(2, 7, 7, 2).astype(np.float32))
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    output_dtype = tf.int64
    include_batch_in_index = False
    name = None

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant(np.random.rand(4, 12, 12, 1).astype(np.float32))
    ksize = [1, 4, 4, 1]
    strides = [1, 4, 4, 1]
    padding = "SAME"
    data_format = "NHWC"
    output_dtype = tf.int32
    include_batch_in_index = True
    name = "pool"

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_tensor = tf.constant(np.random.rand(1, 8, 8, 1).astype(np.float64))
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    output_dtype = tf.int64
    include_batch_in_index = False
    name = None

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = tf.constant(np.random.rand(1, 11, 11, 3).astype(np.float64))
    ksize = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    output_dtype = tf.int32
    include_batch_in_index = True
    name = "max_pool"

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant(np.random.rand(2, 9, 9, 2).astype(np.float64))
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    data_format = "NHWC"
    output_dtype = tf.int64
    include_batch_in_index = False
    name = None

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant(np.random.rand(4, 13, 13, 1).astype(np.float64))
    ksize = [1, 4, 4, 1]
    strides = [1, 4, 4, 1]
    padding = "SAME"
    data_format = "NHWC"
    output_dtype = tf.int32
    include_batch_in_index = True
    name = "pool"

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant(np.random.rand(1, 6, 6, 1).astype(np.int32))
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    data_format = "NHWC"
    output_dtype = tf.int64
    include_batch_in_index = False
    name = None

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant(np.random.rand(1, 12, 12, 3).astype(np.int32))
    ksize = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    data_format = "NHWC"
    output_dtype = tf.int32
    include_batch_in_index = True
    name = "max_pool"

    input_dict = {
        "input": input_tensor.numpy(),
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "data_format": data_format,
        "output_dtype": output_dtype,
        "include_batch_in_index": include_batch_in_index,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool_with_argmax"] = tf_nn_max_pool_with_argmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool_with_argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool_with_argmax'.")

check_valid('tf.nn.max_pool_with_argmax', generated_inputs['tf.nn.max_pool_with_argmax'], lib="tf", suffix=0)
