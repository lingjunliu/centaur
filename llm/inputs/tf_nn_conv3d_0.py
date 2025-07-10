
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 3, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_1"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 10, 5).astype(np.float32)
    filters_tensor = np.random.rand(5, 5, 5, 5, 2).astype(np.float32)
    strides_list = [1, 2, 2, 2, 1]
    padding_string = "SAME"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_2"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 7, 7, 7, 1).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 2, 1, 3).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 2, 2, 2, 1]
    name_string = "conv3d_3"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(3, 8, 8, 8, 4).astype(np.float32)
    filters_tensor = np.random.rand(4, 4, 4, 4, 2).astype(np.float32)
    strides_list = [1, 2, 1, 2, 1]
    padding_string = "SAME"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_4"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 6, 6, 6, 2).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 2, 1).astype(np.float32)
    strides_list = [1, 1, 2, 1, 1]
    padding_string = "VALID"
    data_format_string = "NDHWC"
    dilations_list = [1, 1, 2, 1, 1]
    name_string = "conv3d_5"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 3, 1).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NCDHW"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_6"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(2, 10, 10, 10, 5).astype(np.float32)
    filters_tensor = np.random.rand(5, 5, 5, 5, 2).astype(np.float32)
    strides_list = [1, 2, 2, 2, 1]
    padding_string = "SAME"
    data_format_string = "NCDHW"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_7"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 7, 7, 7, 1).astype(np.float32)
    filters_tensor = np.random.rand(2, 2, 2, 1, 3).astype(np.float32)
    strides_list = [1, 1, 1, 1, 1]
    padding_string = "VALID"
    data_format_string = "NCDHW"
    dilations_list = [1, 2, 2, 2, 1]
    name_string = "conv3d_8"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(3, 8, 8, 8, 4).astype(np.float32)
    filters_tensor = np.random.rand(4, 4, 4, 4, 2).astype(np.float32)
    strides_list = [1, 2, 1, 2, 1]
    padding_string = "SAME"
    data_format_string = "NCDHW"
    dilations_list = [1, 1, 1, 1, 1]
    name_string = "conv3d_9"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 6, 6, 6, 2).astype(np.float32)
    filters_tensor = np.random.rand(3, 3, 3, 2, 1).astype(np.float32)
    strides_list = [1, 1, 2, 1, 1]
    padding_string = "VALID"
    data_format_string = "NCDHW"
    dilations_list = [1, 1, 2, 1, 1]
    name_string = "conv3d_10"
    input_dict = {"input": tf.convert_to_tensor(input_tensor), "filters": tf.convert_to_tensor(filters_tensor), "strides": strides_list, "padding": padding_string, "data_format": data_format_string, "dilations": dilations_list, "name": name_string}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv3d"] = tf_nn_conv3d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv3d'.")

check_valid('tf.nn.conv3d', generated_inputs['tf.nn.conv3d'], lib="tf", suffix=0)
