
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_conv2d_transpose_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(1, 5, 5, 3).astype(np.float32)
    filters1 = np.random.rand(3, 3, 2, 3).astype(np.float32)
    output_shape1 = np.array([1, 7, 7, 2]).astype(np.int32)
    strides1 = [1, 1, 1, 1]
    padding1 = "SAME"
    data_format1 = "NHWC"
    dilations1 = [1, 1, 1, 1]
    name1 = "deconv1"
    input_dict = {"input": input1, "filters": filters1, "output_shape": output_shape1, "strides": strides1, "padding": padding1, "data_format": data_format1, "dilations": dilations1, "name": name1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.conv2d_transpose"] = tf_nn_conv2d_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.conv2d_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.conv2d_transpose'.")

check_valid('tf.nn.conv2d_transpose', generated_inputs['tf.nn.conv2d_transpose'], lib="tf", suffix=0)
