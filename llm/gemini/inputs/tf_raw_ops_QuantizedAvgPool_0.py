
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_quantized_avg_pool_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[[1, 2]], [[3, 4]]]], dtype=np.qint8)
    min_input = np.array(-1.0, dtype=np.float32)
    max_input = np.array(1.0, dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.quint8)
    min_input = np.array(0.0, dtype=np.float32)
    max_input = np.array(255.0, dtype=np.float32)
    ksize = [1, 2, 3, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.qint32)
    min_input = np.array(-100.0, dtype=np.float32)
    max_input = np.array(100.0, dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.qint16)
    min_input = np.array(-50.0, dtype=np.float32)
    max_input = np.array(50.0, dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.quint16)
    min_input = np.array(0.0, dtype=np.float32)
    max_input = np.array(1000.0, dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: different batch size
    input_tensor = np.array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=np.qint8)
    min_input = np.array(-20.0, dtype=np.float32)
    max_input = np.array(20.0, dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.qint32)
    min_input = np.array(-1000.0, dtype=np.float32)
    max_input = np.array(1000.0, dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    input_tensor = np.array([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=np.quint8)
    min_input = np.array(0.0, dtype=np.float32)
    max_input = np.array(255.0, dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.qint8)
    min_input = np.array(-20.0, dtype=np.float32)
    max_input = np.array(20.0, dtype=np.float32)
    ksize = [1, 1, 1, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10
    input_tensor = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.quint8)
    min_input = np.array(0.0, dtype=np.float32)
    max_input = np.array(100.0, dtype=np.float32)
    ksize = [1, 2, 2, 1]
    strides = [1, 1, 1, 1]
    padding = "SAME"

    input_dict = {
        "input": input_tensor,
        "min_input": min_input,
        "max_input": max_input,
        "ksize": ksize,
        "strides": strides,
        "padding": padding,
        "name": "quantized_avg_pool_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.QuantizedAvgPool"] = tf_raw_ops_quantized_avg_pool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedAvgPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAvgPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.QuantizedAvgPool', generated_inputs['tf.raw_ops.QuantizedAvgPool'], lib="tf", suffix=0)
