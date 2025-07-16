
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation1"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.float32)
    filter_tensor = np.array([[[1, 2, 3]]], dtype=np.float32)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation2"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "VALID"
    name = "dilation3"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float64)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.float64)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation4"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.uint8)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation5"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int16)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.int16)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "dilation6"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int8)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.int8)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation7"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int64)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.int64)
    strides = [1, 1, 1, 1]
    rates = [1, 2, 2, 1]
    padding = "SAME"
    name = "dilation8"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float16)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.float16)
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    name = "dilation9"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float16)
    filter_tensor = np.array([[[1, 2], [3, 4]]], dtype=np.float16)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    name = "dilation10"
    input_dict = {"input": input_tensor, "filter": filter_tensor, "strides": strides, "rates": rates, "padding": padding, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dilation2D"] = tf_raw_ops_dilation2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2D'.")

check_valid('tf.raw_ops.Dilation2D', generated_inputs['tf.raw_ops.Dilation2D'], lib="tf", suffix=0)
