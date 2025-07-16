
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_parallel_stack_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D tensors
    values = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    name = "stack_1d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensors
    values = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    name = "stack_2d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float values
    values = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    name = "stack_float"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values
    values = [np.array([-1, -2, -3]), np.array([-4, -5, -6])]
    name = "stack_negative"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean values
    values = [np.array([True, False, True]), np.array([False, True, False])]
    name = "stack_bool"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Int64 values
    values = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)]
    name = "stack_int64"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Float64 values
    values = [np.array([1.0, 2.0, 3.0], dtype=np.float64), np.array([4.0, 5.0, 6.0], dtype=np.float64)]
    name = "stack_float64"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensors
    values = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    name = "stack_3d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex 2D array with different data type
    values = [np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32), np.array([[5.5, 6.6], [7.7, 8.8]], dtype=np.float32)]
    name = "stack_complex_2d"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: single value arrays
    values = [np.array([1]), np.array([2])]
    name = "stack_single_val"
    input_dict = {"values": values, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.parallel_stack"] = tf_parallel_stack_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.parallel_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.parallel_stack'.")

check_valid('tf.parallel_stack', generated_inputs['tf.parallel_stack'], lib="tf", suffix=0)
