
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_raw_ops_densebincount_inputs():
    list_of_inputs = []

    # Input 1: Basic case with int32 input and weights
    input_arr = np.array([1, 2, 3, 4, 5, 0, 1, 2], dtype=np.int32)
    size_val = np.array(6, dtype=np.int32)
    weights_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 2: int64 input and weights
    input_arr = np.array([1, 2, 3, 4, 5, 0, 1, 2], dtype=np.int64)
    size_val = np.array(6, dtype=np.int64)
    weights_arr = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 3: Empty weights (acts as all weights equal to 1)
    input_arr = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int32)
    size_val = np.array(6, dtype=np.int32)
    weights_arr = np.array([], dtype=np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 4: 2D input
    input_arr = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int32)
    size_val = np.array(4, dtype=np.int32)
    weights_arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 5: Weights with float64
    input_arr = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    size_val = np.array(7, dtype=np.int32)
    weights_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 6: Size larger than max value in input
    input_arr = np.array([0, 1, 2], dtype=np.int32)
    size_val = np.array(5, dtype=np.int32)
    weights_arr = np.array([1, 2, 3], dtype=np.int32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 7: Size equals max value in input + 1
    input_arr = np.array([0, 1, 2], dtype=np.int32)
    size_val = np.array(3, dtype=np.int32)
    weights_arr = np.array([1, 2, 3], dtype=np.int32)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 8: int64 weights with int32 input
    input_arr = np.array([0, 1, 2], dtype=np.int32)
    size_val = np.array(4, dtype=np.int32)
    weights_arr = np.array([1, 2, 3], dtype=np.int64)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 9: Different values in input, binary_output true
    input_arr = np.array([1, 3, 5, 2, 4], dtype=np.int32)
    size_val = np.array(6, dtype=np.int32)
    weights_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    binary_output_val = True
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    # Input 10: Input with repeated values and float weights
    input_arr = np.array([1, 2, 2, 3, 1, 0], dtype=np.int32)
    size_val = np.array(4, dtype=np.int32)
    weights_arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    binary_output_val = False
    name_val = None

    input_dict = {
        "input": tf.convert_to_tensor(input_arr),
        "size": tf.convert_to_tensor(size_val),
        "weights": tf.convert_to_tensor(weights_arr),
        "binary_output": binary_output_val,
        "name": name_val
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DenseBincount"] = tf_raw_ops_densebincount_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DenseBincount' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DenseBincount'.")

check_valid('tf.raw_ops.DenseBincount', generated_inputs['tf.raw_ops.DenseBincount'], lib="tf", suffix=0)
