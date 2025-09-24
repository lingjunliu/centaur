
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = tf.constant(np.random.rand(1, 5, 1).astype(np.float32)).numpy()
    ksize = [2]
    strides = [2]
    padding = 'VALID'
    data_format = 'NWC'
    name = 'avgpool1d_1'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = tf.constant(np.random.rand(1, 10, 3).astype(np.float32)).numpy()
    ksize = [3]
    strides = [1]
    padding = 'SAME'
    data_format = 'NWC'
    name = 'avgpool1d_2'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = tf.constant(np.random.rand(2, 8, 2).astype(np.float32)).numpy()
    ksize = [4]
    strides = [2]
    padding = 'VALID'
    data_format = 'NWC'
    name = 'avgpool1d_3'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = tf.constant(np.random.rand(1, 12, 1).astype(np.float32)).numpy()
    ksize = [5]
    strides = [3]
    padding = 'SAME'
    data_format = 'NWC'
    name = 'avgpool1d_4'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = tf.constant(np.random.rand(1, 7, 1).astype(np.float32)).numpy()
    ksize = [1]
    strides = [1]
    padding = 'VALID'
    data_format = 'NWC'
    name = 'avgpool1d_5'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6
    input_tensor = tf.constant(np.random.rand(1, 7, 1).astype(np.float32)).numpy()
    ksize = [1]
    strides = [1]
    padding = 'SAME'
    data_format = 'NWC'
    name = 'avgpool1d_6'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = tf.constant(np.random.rand(3, 15, 4).astype(np.float32)).numpy()
    ksize = [5]
    strides = [5]
    padding = 'VALID'
    data_format = 'NWC'
    name = 'avgpool1d_7'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = tf.constant(np.random.rand(1, 6, 1).astype(np.float32)).numpy()
    ksize = [2]
    strides = [1]
    padding = 'SAME'
    data_format = 'NWC'
    name = 'avgpool1d_8'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = tf.constant(np.random.rand(1, 5, 1).astype(np.float32)).numpy()
    ksize = [3]
    strides = [1]
    padding = 'VALID'
    data_format = 'NWC'
    name = 'avgpool1d_9'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = tf.constant(np.random.rand(1, 10, 1).astype(np.float32)).numpy()
    ksize = [3]
    strides = [2]
    padding = 'SAME'
    data_format = 'NWC'
    name = 'avgpool1d_10'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11, NCW format
    input_tensor = tf.constant(np.random.rand(1, 1, 5).astype(np.float32)).numpy()
    ksize = [2]
    strides = [2]
    padding = 'VALID'
    data_format = 'NCW'
    name = 'avgpool1d_11'
    input_dict = {'input': input_tensor, 'ksize': ksize, 'strides': strides, 'padding': padding, 'data_format': data_format, 'name': name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.avg_pool1d"] = tf_nn_avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.avg_pool1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool1d'.")

check_valid('tf.nn.avg_pool1d', generated_inputs['tf.nn.avg_pool1d'], lib="tf", suffix=0)
