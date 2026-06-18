
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_avg_pool_inputs():
    list_of_inputs = []
    
    # Input 1: N=2, NHWC, 2x2 pool, 2x2 stride
    input_1 = np.random.randn(2, 8, 8, 3).astype(np.float32)
    ksize_1 = [2, 2]
    strides_1 = [2, 2]
    padding_1 = "VALID"
    data_format_1 = "NHWC"
    name_1 = "pool1"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_1,
        "ksize": ksize_1,
        "strides": strides_1,
        "padding": padding_1,
        "data_format": data_format_1,
        "name": name_1
    }))
    
    # Input 2: N=2, NCHW, SAME padding, length N+2 ksize
    input_2 = np.random.randn(2, 3, 8, 8).astype(np.float32)
    ksize_2 = [1, 1, 2, 2]
    strides_2 = [1, 1, 2, 2]
    padding_2 = "SAME"
    data_format_2 = "NCHW"
    name_2 = "pool2"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_2,
        "ksize": ksize_2,
        "strides": strides_2,
        "padding": padding_2,
        "data_format": data_format_2,
        "name": name_2
    }))
    
    # Input 3: N=1, NWC, 2 pool, 1 stride
    input_3 = np.random.randn(1, 10, 4).astype(np.float32)
    ksize_3 = [2]
    strides_3 = [1]
    padding_3 = "VALID"
    data_format_3 = "NWC"
    name_3 = "pool3"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_3,
        "ksize": ksize_3,
        "strides": strides_3,
        "padding": padding_3,
        "data_format": data_format_3,
        "name": name_3
    }))

    # Input 4: N=1, NCW, SAME padding, length N+2 ksize
    input_4 = np.random.randn(1, 4, 10).astype(np.float32)
    ksize_4 = [1, 1, 2]
    strides_4 = [1, 1, 1]
    padding_4 = "SAME"
    data_format_4 = "NCW"
    name_4 = "pool4"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_4,
        "ksize": ksize_4,
        "strides": strides_4,
        "padding": padding_4,
        "data_format": data_format_4,
        "name": name_4
    }))

    # Input 5: N=3, NDHWC, 2x2x2 pool, 2x2x2 stride
    input_5 = np.random.randn(2, 4, 4, 4, 3).astype(np.float32)
    ksize_5 = [2, 2, 2]
    strides_5 = [2, 2, 2]
    padding_5 = "VALID"
    data_format_5 = "NDHWC"
    name_5 = "pool5"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_5,
        "ksize": ksize_5,
        "strides": strides_5,
        "padding": padding_5,
        "data_format": data_format_5,
        "name": name_5
    }))

    # Input 6: N=3, NCDHW, SAME padding, length N+2 ksize
    input_6 = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)
    ksize_6 = [1, 1, 2, 2, 2]
    strides_6 = [1, 1, 2, 2, 2]
    padding_6 = "SAME"
    data_format_6 = "NCDHW"
    name_6 = "pool6"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_6,
        "ksize": ksize_6,
        "strides": strides_6,
        "padding": padding_6,
        "data_format": data_format_6,
        "name": name_6
    }))

    # Input 7: N=2, NHWC, float64
    input_7 = np.random.randn(1, 6, 6, 2).astype(np.float64)
    ksize_7 = [3, 3]
    strides_7 = [1, 1]
    padding_7 = "VALID"
    data_format_7 = "NHWC"
    name_7 = "pool7"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_7,
        "ksize": ksize_7,
        "strides": strides_7,
        "padding": padding_7,
        "data_format": data_format_7,
        "name": name_7
    }))

    # Input 8: N=2, NHWC, length 1 ksize
    input_8 = np.random.uniform(-10.0, 10.0, (2, 5, 5, 2)).astype(np.float32)
    ksize_8 = [1]
    strides_8 = [1]
    padding_8 = "SAME"
    data_format_8 = "NHWC"
    name_8 = "pool8"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_8,
        "ksize": ksize_8,
        "strides": strides_8,
        "padding": padding_8,
        "data_format": data_format_8,
        "name": name_8
    }))

    # Input 9: N=2, NCHW, stride larger than pool size
    input_9 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    ksize_9 = [2, 2]
    strides_9 = [3, 3]
    padding_9 = "VALID"
    data_format_9 = "NCHW"
    name_9 = "pool9"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_9,
        "ksize": ksize_9,
        "strides": strides_9,
        "padding": padding_9,
        "data_format": data_format_9,
        "name": name_9
    }))

    # Input 10: N=1, NWC, large strides
    input_10 = np.random.randn(4, 20, 1).astype(np.float32)
    ksize_10 = [5]
    strides_10 = [5]
    padding_10 = "SAME"
    data_format_10 = "NWC"
    name_10 = "pool10"
    
    list_of_inputs.append(copy.deepcopy({
        "input": input_10,
        "ksize": ksize_10,
        "strides": strides_10,
        "padding": padding_10,
        "data_format": data_format_10,
        "name": name_10
    }))

    return list_of_inputs

generated_inputs["tf.nn.avg_pool"] = tf_nn_avg_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.avg_pool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.avg_pool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.avg_pool', generated_inputs['tf.nn.avg_pool'], lib="tf", suffix=0)
