
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tfenp_meshgrid_inputs():
    """
    Generates a list of valid inputs for tf.experimental.numpy.meshgrid.
    """
    list_of_inputs = []

    # Input 1: Two simple 1D integer arrays of the same length
    input_dict_1 = {
        'xi': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Three 1D integer arrays of the same length
    input_dict_2 = {
        'xi': np.array([np.arange(4, dtype=np.int64), np.arange(10, 14, dtype=np.int64), np.arange(20, 24, dtype=np.int64)])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Two 1D float arrays of the same length
    input_dict_3 = {
        'xi': np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Arrays with negative values and zero, same length
    input_dict_4 = {
        'xi': np.array([[-1, -2, -3], [1, 0, -1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A single 1D input array
    input_dict_5 = {
        'xi': np.array([[10, 20, 30, 40]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Four 1D input arrays, length 1
    input_dict_6 = {
        'xi': np.array([[1], [2], [3], [4]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Two 1D arrays of different float types (will be promoted)
    input_dict_7 = {
        'xi': np.array([[1, 2], [3.0, 4.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Two 1D arrays that are empty
    input_dict_8 = {
        'xi': np.empty(shape=(2, 0), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A single empty array
    input_dict_9 = {
        'xi': np.empty(shape=(1, 0), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Two 1D arrays with single elements
    input_dict_10 = {
        'xi': np.array([[100.0], [-100.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: A larger number of arrays
    input_dict_11 = {
        'xi': np.array([[1,1], [2,2], [3,3], [4,4], [5,5]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Long arrays
    input_dict_12 = {
        'xi': np.array([np.arange(10), np.arange(10, 20)], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.meshgrid"] = tfenp_meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.meshgrid'.")

check_valid('tf.experimental.numpy.meshgrid', generated_inputs['tf.experimental.numpy.meshgrid'], lib="tf", suffix=0)
