
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MaxPool3D_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_1"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 10, 10, 10, 1).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "maxpool3d_2"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 8, 8, 8, 3).astype(np.float32)
    ksize = [1, 4, 4, 4, 1]
    strides = [1, 3, 3, 3, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_3"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 6, 6, 6, 5).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "maxpool3d_4"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 7, 7, 7, 1).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_5"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 3, 3, 3, 1).astype(np.float32)
    ksize = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_6"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 5, 5, 5, 3).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "maxpool3d_7"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 NCDHW
    input_tensor = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    ksize = [1, 1, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_8"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 4, 4, 4, 2).astype(np.float32)
    ksize = [1, 2, 2, 2, 1]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"
    data_format = "NDHWC"
    name = "maxpool3d_9"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.random.rand(1, 6, 6, 6, 1).astype(np.float32)
    ksize = [1, 3, 3, 3, 1]
    strides = [1, 2, 2, 2, 1]
    padding = "VALID"
    data_format = "NDHWC"
    name = "maxpool3d_10"
    input_dict = {"input": input_tensor, "ksize": ksize, "strides": strides, "padding": padding, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MaxPool3D"] = tf_raw_ops_MaxPool3D_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MaxPool3D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MaxPool3D'.")

check_valid('tf.raw_ops.MaxPool3D', generated_inputs['tf.raw_ops.MaxPool3D'], lib="tf", suffix=0)
