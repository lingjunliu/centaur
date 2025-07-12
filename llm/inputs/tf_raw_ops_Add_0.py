
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_add_inputs():
    list_of_inputs = []

    # Input 1: Basic addition of two matrices
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[5, 6], [7, 8]], dtype=np.int32)
    name = "add_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Addition with negative values
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int32)
    y = np.array([[5, -6], [7, -8]], dtype=np.int32)
    name = "add_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Addition of two 1D arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    name = "add_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Addition with floating point numbers
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    y = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float32)
    name = "add_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Addition with complex numbers
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    y = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex64)
    name = "add_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Addition with different shapes (broadcasting)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([5, 6], dtype=np.int32)
    name = "add_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Addition with uint8 type
    x = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    y = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    name = "add_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Addition with int8 type and negative numbers
    x = np.array([[-1, 2], [3, -4]], dtype=np.int8)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int8)
    name = "add_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Addition with rank 3 tensor
    x = np.random.randint(0, 10, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(0, 10, size=(2, 3, 4), dtype=np.int32)
    name = "add_example_9"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Addition with float64 type
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    y = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float64)
    name = "add_example_10"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Add"] = tf_raw_ops_add_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Add' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Add'.")

check_valid('tf.raw_ops.Add', generated_inputs['tf.raw_ops.Add'], lib="tf", suffix=0)
