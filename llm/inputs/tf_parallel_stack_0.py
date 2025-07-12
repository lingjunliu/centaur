
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_parallel_stack_inputs():
    list_of_inputs = []

    def convert_to_numpy_arrays(values):
      return [np.array(val) for val in values]

    # Input 1: Simple 1D arrays
    values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    values = convert_to_numpy_arrays(values)
    input_dict = {"values": values, "name": "stack_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays
    values = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    values = convert_to_numpy_arrays(values)
    input_dict = {"values": values, "name": "stack_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different numerical types (int32)
    values = [[1, 2, 3], [4, 5, 6]]
    values = convert_to_numpy_arrays(values)
    values = [arr.astype(np.int32) for arr in values]
    input_dict = {"values": values, "name": "stack_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different numerical types (float32)
    values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    values = convert_to_numpy_arrays(values)
    values = [arr.astype(np.float32) for arr in values]
    input_dict = {"values": values, "name": "stack_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: different name
    values = [[1, 2], [3, 4]]
    values = convert_to_numpy_arrays(values)
    input_dict = {"values": values, "name": "another_stack"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    values = [[-1, -2], [-3, -4]]
    values = convert_to_numpy_arrays(values)
    input_dict = {"values": values, "name": "negative_stack"}
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
