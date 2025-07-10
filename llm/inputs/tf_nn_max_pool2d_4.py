
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1, valid: basic example
    input1 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize1 = [1, 2, 2, 1]
    strides1 = [1, 2, 2, 1]
    padding1 = 'VALID'
    data_format1 = 'NHWC'
    name1 = 'maxpool1'

    input_dict1 = {
        "input": input1,
        "ksize": ksize1,
        "strides": strides1,
        "padding": padding1,
        "data_format": data_format1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2, valid: SAME padding
    input2 = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    ksize2 = [1, 2, 2, 1]
    strides2 = [1, 1, 1, 1]
    padding2 = 'SAME'
    data_format2 = 'NHWC'
    name2 = 'maxpool2'

    input_dict2 = {
        "input": input2,
        "ksize": ksize2,
        "strides": strides2,
        "padding": padding2,
        "data_format": data_format2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.max_pool2d_4"] = []
generated_inputs["tf.nn.max_pool2d_4"] = tf_nn_max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.max_pool2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.max_pool2d_4'.")

check_valid('tf.nn.max_pool2d', generated_inputs['tf.nn.max_pool2d_4'], lib="tf", suffix=4)
