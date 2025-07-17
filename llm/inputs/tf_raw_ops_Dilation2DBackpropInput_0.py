
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_dilation2d_backprop_input_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2]]]], dtype=np.float32)
    filter_tensor = np.array([[[1, 2]]], dtype=np.float32)
    out_backprop_tensor = np.array([[[[1, 2]]]], dtype=np.float32)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "VALID"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3]]]], dtype=np.float64)
    filter_tensor = np.array([[[1, 2, 3]]], dtype=np.float64)
    out_backprop_tensor = np.array([[[[1, 2, 3]]]], dtype=np.float64)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "SAME"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    filter_tensor = np.array([[[1, 2]]], dtype=np.int32)
    out_backprop_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "VALID"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint8)
    filter_tensor = np.array([[[1, 2]]], dtype=np.uint8)
    out_backprop_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.uint8)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "SAME"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    input_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)
    filter_tensor = np.array([[[1, 2]]], dtype=np.int16)
    out_backprop_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "VALID"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]]] , dtype=np.int8)
    filter_tensor = np.array([[[1, 2, 3]]], dtype=np.int8)
    out_backprop_tensor = np.array([[[[1, 2, 3]]], [[[4, 5, 6]]]], dtype=np.int8)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "SAME"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, 2]]]], dtype=np.int64)
    filter_tensor = np.array([[[1, 2]]], dtype=np.int64)
    out_backprop_tensor = np.array([[[[1, 2]]]], dtype=np.int64)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "VALID"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[[1.0, 2.0]]]], dtype=np.float16)
    filter_tensor = np.array([[[1.0, 2.0]]], dtype=np.float16)
    out_backprop_tensor = np.array([[[[1.0, 2.0]]]], dtype=np.float16)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "VALID"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    filter_tensor = np.array([[[1, 2]]], dtype=np.int32)
    out_backprop_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.int32)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "VALID"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.half)
    filter_tensor = np.array([[[1, 2]]], dtype=np.half)
    out_backprop_tensor = np.array([[[[1, 2], [3, 4]]]], dtype=np.half)
    strides_list = [1, 1, 1, 1]
    rates_list = [1, 1, 1, 1]
    padding_string = "SAME"

    input_dict = {
        "strides": strides_list,
        "rates": rates_list,
        "padding": padding_string,
        "name": None,
        "input": input_tensor,
        "filter": filter_tensor,
        "out_backprop": out_backprop_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Dilation2DBackpropInput"] = tf_raw_ops_dilation2d_backprop_input_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Dilation2DBackpropInput' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Dilation2DBackpropInput'.")

check_valid('tf.raw_ops.Dilation2DBackpropInput', generated_inputs['tf.raw_ops.Dilation2DBackpropInput'], lib="tf", suffix=0)
