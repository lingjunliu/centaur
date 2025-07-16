
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_meshgrid_inputs():
    list_of_inputs = []

    def create_input_dict(x, y, indexing, name):
        return {
            'args': [tf.constant(x), tf.constant(y)],
            'indexing': indexing,
            'name': name
        }

    # Input 1, valid
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_1')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    input_dict = create_input_dict(x, y, 'ij', 'meshgrid_example_2')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid - single input
    x = np.array([1, 2, 3])
    input_dict = {
        'args': [tf.constant(x)],
        'indexing': 'xy',
        'name': 'meshgrid_example_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid - three inputs
    x = np.array([1, 2])
    y = np.array([3, 4])
    z = np.array([5, 6])
    input_dict = {
        'args': [tf.constant(x), tf.constant(y), tf.constant(z)],
        'indexing': 'xy',
        'name': 'meshgrid_example_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - different data types
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_5')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - negative values
    x = np.array([-1, 0, 1])
    y = np.array([-2, 0, 2])
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_6')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid - large values
    x = np.array([1000, 2000, 3000])
    y = np.array([4000, 5000, 6000])
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_7')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - small values
    x = np.array([0.001, 0.002, 0.003])
    y = np.array([0.004, 0.005, 0.006])
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_8')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid - int64
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4, 5, 6], dtype=np.int64)
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_9')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid - different lengths
    x = np.array([1, 2])
    y = np.array([3, 4, 5])
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_10')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: ij indexing, float64
    x = np.array([1.1, 2.2], dtype=np.float64)
    y = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    input_dict = create_input_dict(x, y, 'ij', 'meshgrid_example_11')
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty arrays
    x = np.array([])
    y = np.array([])
    input_dict = create_input_dict(x, y, 'xy', 'meshgrid_example_12')
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.meshgrid"] = tf_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.meshgrid'.")

check_valid('tf.meshgrid', generated_inputs['tf.meshgrid'], lib="tf", suffix=0)
