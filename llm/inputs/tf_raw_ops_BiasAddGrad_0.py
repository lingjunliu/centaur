
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_bias_add_grad_inputs():
    list_of_inputs = []

    # Input 1: Simple float32, NHWC
    out_backprop = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data_format = "NHWC"
    name = "bias_add_grad_1"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple float32, NCHW
    out_backprop = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data_format = "NCHW"
    name = "bias_add_grad_2"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64, NHWC
    out_backprop = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    data_format = "NHWC"
    name = "bias_add_grad_3"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int32, NCHW
    out_backprop = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    data_format = "NCHW"
    name = "bias_add_grad_4"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64, NHWC
    out_backprop = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    data_format = "NHWC"
    name = "bias_add_grad_5"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int8, NCHW
    out_backprop = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int8)
    data_format = "NCHW"
    name = "bias_add_grad_6"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bfloat16, NHWC
    out_backprop = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16).astype(np.float32).astype(np.float16)
    data_format = "NHWC"
    name = "bias_add_grad_7"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Half, NCHW
    out_backprop = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    data_format = "NCHW"
    name = "bias_add_grad_8"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple dimensions, NHWC
    out_backprop = np.random.rand(2, 3, 4, 5).astype(np.float32)
    data_format = "NHWC"
    name = "bias_add_grad_9"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple dimensions, NCHW
    out_backprop = np.random.rand(2, 3, 4, 5).astype(np.float32)
    data_format = "NCHW"
    name = "bias_add_grad_10"
    input_dict = {"out_backprop": out_backprop, "data_format": data_format, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BiasAddGrad"] = tf_raw_ops_bias_add_grad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BiasAddGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BiasAddGrad'.")

check_valid('tf.raw_ops.BiasAddGrad', generated_inputs['tf.raw_ops.BiasAddGrad'], lib="tf", suffix=0)
