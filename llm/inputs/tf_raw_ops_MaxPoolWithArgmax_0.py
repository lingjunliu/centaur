
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPoolWithArgmax_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    Targmax = tf.int64
    include_batch_in_index = False
    name = "max_pool_1"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]).astype(np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    Targmax = tf.int32
    include_batch_in_index = True
    name = "max_pool_2"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    Targmax = tf.int64
    include_batch_in_index = False
    name = "max_pool_3"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float64)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    Targmax = tf.int64
    include_batch_in_index = True
    name = "max_pool_4"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]]]]).astype(np.int32)
    ksize = [1, 1, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    Targmax = tf.int32
    include_batch_in_index = False
    name = "max_pool_5"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2], [3, 4]]]]).astype(np.uint8)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"
    Targmax = tf.int64
    include_batch_in_index = True
    name = "max_pool_6"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.int64)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"
    Targmax = tf.int32
    include_batch_in_index = False
    name = "max_pool_7"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.half)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    Targmax = tf.int64
    include_batch_in_index = True
    name = "max_pool_8"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.uint32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    Targmax = tf.int32
    include_batch_in_index = False
    name = "max_pool_9"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.uint64)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"
    Targmax = tf.int64
    include_batch_in_index = True
    name = "max_pool_10"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "Targmax": Targmax, "include_batch_in_index": include_batch_in_index, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPoolWithArgmax"] = tf_raw_ops_MaxPoolWithArgmax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPoolWithArgmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPoolWithArgmax'.")

check_valid('tf.raw_ops.MaxPoolWithArgmax', generated_inputs['tf.raw_ops.MaxPoolWithArgmax'], lib="tf", suffix=0)
