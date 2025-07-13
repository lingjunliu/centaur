
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_FakeQuantWithMinMaxVars_inputs():
    list_of_inputs = []

    # Input 1
    inputs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    min_val = np.array(-1.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test1"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    inputs = np.array([[1.2, -0.3, 0.7], [2.1, 0.5, -1.0]], dtype=np.float32)
    min_val = np.array(-0.5, dtype=np.float32)
    max_val = np.array(0.8, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test2"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 4
    narrow_range = True
    name = "test3"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    inputs = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32)
    min_val = np.array(-2.0, dtype=np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    num_bits = 16
    narrow_range = False
    name = "test4"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    inputs = np.array([1.0], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    num_bits = 2
    narrow_range = True
    name = "test5"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    inputs = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    min_val = np.array(1.0, dtype=np.float32)
    max_val = np.array(8.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test6"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    inputs = np.array([-0.5], dtype=np.float32)
    min_val = np.array(-1.0, dtype=np.float32)
    max_val = np.array(0.0, dtype=np.float32)
    num_bits = 8
    narrow_range = True
    name = "test7"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    inputs = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(20.0, dtype=np.float32)
    num_bits = 12
    narrow_range = False
    name = "test8"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    inputs = np.array([[-0.1, 0.2], [-0.3, 0.4]], dtype=np.float32)
    min_val = np.array(-0.5, dtype=np.float32)
    max_val = np.array(0.5, dtype=np.float32)
    num_bits = 6
    narrow_range = True
    name = "test9"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    inputs = np.array([0.0], dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(0.0, dtype=np.float32)
    num_bits = 8
    narrow_range = False
    name = "test10"

    input_dict = {
        "inputs": tf.convert_to_tensor(inputs, dtype=tf.float32),
        "min": tf.convert_to_tensor(min_val, dtype=tf.float32),
        "max": tf.convert_to_tensor(max_val, dtype=tf.float32),
        "num_bits": num_bits,
        "narrow_range": narrow_range,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVars"] = tf_raw_ops_FakeQuantWithMinMaxVars_inputs()
tf.experimental.numpy.experimental_enable_numpy_behavior()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVars' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVars'.")

check_valid('tf.raw_ops.FakeQuantWithMinMaxVars', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVars'], lib="tf", suffix=0)
