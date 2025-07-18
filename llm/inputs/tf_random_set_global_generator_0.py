
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_random_set_global_generator_inputs():
    """
    Generates inputs for tf.random.set_global_generator.

    NOTE: The API expects a `tf.random.Generator` object. However, the provided
    signature is `{'generator': 'tensor'}` and the instructions strictly require
    adherence to this signature and the use of numpy format for tensors.
    Therefore, the following inputs are numpy arrays that could be interpreted
    as either a seed (if scalar) or a state (if a vector) from which a
    `tf.random.Generator` object could be created by a calling framework.
    A state for 'philox' algorithm is a shape (3,) int64 tensor.
    A state for 'threefry' algorithm is a shape (2,) int64 tensor.
    """
    list_of_inputs = []

    # Input 1: A potential state for 'philox' algorithm
    input_dict1 = {'generator': np.array([123, 456, 789], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: A potential state for 'threefry' algorithm
    input_dict2 = {'generator': np.array([987, 654], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: A potential state for 'philox' with all zeros
    input_dict3 = {'generator': np.array([0, 0, 0], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: A potential state for 'threefry' with all zeros
    input_dict4 = {'generator': np.array([0, 0], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: A potential state for 'philox' with large integer values
    input_dict5 = {'generator': np.array([2**31 - 1, 2**32, 9000000000000000000], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: A potential state for 'threefry' with large integer values (corrected)
    input_dict6 = {'generator': np.array([1234567890123456789, 8765432109876543210], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: A potential state for 'philox' with negative values
    input_dict7 = {'generator': np.array([-1, -100, -200], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: A potential state for 'threefry' with negative values
    input_dict8 = {'generator': np.array([-12345, -67890], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: A potential state for 'philox' with mixed-sign values
    input_dict9 = {'generator': np.array([100, -200, 300], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: A potential state for 'threefry' with mixed-sign values
    input_dict10 = {'generator': np.array([500, -500], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    # Input 11: A potential seed (scalar tensor)
    input_dict11 = {'generator': np.array(42, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: A potential zero seed (scalar tensor)
    input_dict12 = {'generator': np.array(0, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs["tf.random.set_global_generator"] = tf_random_set_global_generator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.set_global_generator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.set_global_generator'.")

check_valid('tf.random.set_global_generator', generated_inputs['tf.random.set_global_generator'], lib="tf", suffix=0)
